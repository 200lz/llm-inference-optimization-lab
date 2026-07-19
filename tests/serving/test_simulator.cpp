#include "serving/checked_math.h"
#include "serving/event.h"
#include "serving/request.h"
#include "serving/simulated_backend.h"
#include "serving/simulation_clock.h"
#include "serving/simulation_engine.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace llm_lab::serving::test {

struct SimulationEngineTestAccess {
  static void inject_event(SimulationEngine& engine,
                           std::int64_t timestamp_us, RequestId request_id,
                           EventType type) {
    (void)engine.events_.schedule(timestamp_us, request_id, type);
  }
};

}  // namespace llm_lab::serving::test

namespace {

using llm_lab::serving::Event;
using llm_lab::serving::EventQueue;
using llm_lab::serving::EventType;
using llm_lab::serving::Backend;
using llm_lab::serving::Request;
using llm_lab::serving::RequestId;
using llm_lab::serving::RequestState;
using llm_lab::serving::SimulatedBackend;
using llm_lab::serving::SimulationEngine;
using llm_lab::serving::SimulationEngineConfig;

static_assert(
    std::is_constructible_v<SimulationEngine, const SimulatedBackend&>,
    "SimulationEngine must accept a named backend lvalue");
static_assert(
    !std::is_constructible_v<SimulationEngine, SimulatedBackend&&>,
    "SimulationEngine must reject a temporary backend");
static_assert(
    !std::is_constructible_v<SimulationEngine, SimulatedBackend&&,
                             SimulationEngineConfig>,
    "configured SimulationEngine must reject a temporary backend");

class ThrowingDecodeBackend final : public Backend {
 public:
  void validate_configuration() const override {}

  std::int64_t estimate_prefill_us(std::uint64_t,
                                   std::uint64_t) const override {
    return 1;
  }

  std::int64_t estimate_decode_step_us(std::uint64_t) const override {
    throw std::runtime_error("deterministic decode backend failure");
  }
};

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

SimulatedBackend standard_backend() {
  return SimulatedBackend({10, 2, 1, 4, 1});
}

void test_single_request_lifecycle_and_time_zero() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 3, 2));
  engine.run();

  const Request& request = engine.request({1});
  require(request.state == RequestState::Finished, "request did not finish");
  require(request.arrival_time_us == 0, "arrival at time zero changed");
  require(request.admitted_time_us.has_value(), "request has no admission time");
  require(*request.admitted_time_us == 0, "request was not admitted at zero");
  require(request.first_scheduled_time_us.has_value(),
          "request has no first scheduled time");
  require(*request.first_scheduled_time_us == 0,
          "request was not first scheduled at zero");
  require(request.first_token_time_us.has_value(),
          "request has no first-token time");
  require(*request.first_token_time_us == 22,
          "unexpected first-token timestamp");
  require(request.finish_time_us.has_value(), "request has no finish time");
  require(*request.finish_time_us == 27, "unexpected finish timestamp");
  require(request.generated_token_count == 2, "wrong generated token count");
  require(!engine.failed(), "successful engine reported failure");
  require(engine.processed_event_count() == engine.event_log().size(),
          "successful event count disagrees with event log");
  require(engine.event_queue_empty(), "event queue did not drain");
}

void test_zero_prompt_and_zero_output() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 0, 1));
  engine.submit_request(Request({2}, 0, 5, 0));
  engine.run();

  const Request& empty_prompt = engine.request({1});
  require(empty_prompt.generated_token_count == 1,
          "zero-length prompt did not decode");
  const Request& empty_output = engine.request({2});
  require(empty_output.state == RequestState::Finished,
          "zero-output request did not finish");
  require(empty_output.generated_token_count == 0,
          "zero-output request generated a token");
  require(!empty_output.first_token_time_us.has_value(),
          "zero-output request has a first-token timestamp");
  require(empty_output.finish_time_us.has_value(),
          "zero-output request has no finish timestamp");
  require(*empty_output.finish_time_us == 37,
          "zero-output request finished at the wrong time");

  std::vector<Event> empty_output_trace;
  for (const Event& event : engine.event_log()) {
    if (event.request_id == RequestId{2}) {
      empty_output_trace.push_back(event);
    }
  }
  require(empty_output_trace.size() == 3,
          "zero-output request has an unexpected event count");
  require(empty_output_trace[0].type == EventType::RequestArrival &&
              empty_output_trace[1].type == EventType::PrefillComplete &&
              empty_output_trace[2].type == EventType::RequestComplete,
          "zero-output request has an unexpected event trace");
  require(empty_output_trace[1].timestamp_us == 37 &&
              empty_output_trace[2].timestamp_us == 37,
          "zero-output completion did not match prefill completion time");
  for (const Event& event : empty_output_trace) {
    require(event.type != EventType::DecodeStepComplete,
            "zero-output request has a decode event");
  }
}

void test_first_token_assigned_once() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 1, 3));
  engine.run();

  const auto& request = engine.request({1});
  require(request.first_token_time_us.has_value(),
          "multi-token request has no first-token time");
  require(*request.first_token_time_us == 18,
          "first-token timestamp was overwritten by a later decode step");
  require(request.finish_time_us.has_value(),
          "multi-token request has no finish time");
  require(*request.finish_time_us == 28,
          "unexpected multi-token finish time");
}

void test_event_ordering_is_stable() {
  EventQueue queue;
  queue.schedule(5, {3}, EventType::DecodeStepComplete);
  queue.schedule(2, {9}, EventType::RequestArrival);
  queue.schedule(5, {1}, EventType::RequestComplete);
  require(queue.pop().request_id == RequestId{9}, "timestamp ordering failed");
  require(queue.pop().request_id == RequestId{3},
          "equal-time insertion ordering failed");
  require(queue.pop().request_id == RequestId{1},
          "equal-time insertion ordering used event type or request ID");
}

void test_clock_is_monotonic() {
  llm_lab::serving::SimulationClock clock;
  require(clock.now_us() == 0, "clock did not start at zero");
  clock.advance_to(0);
  clock.advance_to(7);
  require_throws<std::invalid_argument>([&] { clock.advance_to(6); },
                                        "clock moved backwards");
}

std::vector<Event> deterministic_trace() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({4}, 2, 3, 2));
  engine.submit_request(Request({2}, 2, 0, 0));
  engine.submit_request(Request({8}, 9, 1, 1));
  engine.run();
  return engine.event_log();
}

void test_repeated_execution_is_deterministic() {
  const auto lhs = deterministic_trace();
  const auto rhs = deterministic_trace();
  require(lhs.size() == rhs.size(), "trace sizes differ");
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    require(lhs[i].timestamp_us == rhs[i].timestamp_us &&
                lhs[i].event_sequence == rhs[i].event_sequence &&
                lhs[i].request_id == rhs[i].request_id &&
                lhs[i].type == rhs[i].type,
            "repeated event traces differ");
  }
}

void test_state_transitions() {
  Request request({1}, 0, 1, 1);
  request.transition_to(RequestState::Prefilling);
  request.transition_to(RequestState::Preempted);
  request.transition_to(RequestState::Waiting);
  request.transition_to(RequestState::Cancelled);
  require_throws<std::logic_error>(
      [&] { request.transition_to(RequestState::Prefilling); },
      "cancelled request returned to an active state");

  Request finished({2}, 0, 1, 1);
  require_throws<std::logic_error>(
      [&] { finished.transition_to(RequestState::Finished); },
      "invalid Waiting-to-Finished transition succeeded");
  finished.transition_to(RequestState::Prefilling);
  require_throws<std::logic_error>(
      [&] { finished.transition_to(RequestState::Finished); },
      "positive-output request skipped decoding");

  Request no_output({3}, 0, 1, 0);
  no_output.transition_to(RequestState::Prefilling);
  no_output.transition_to(RequestState::Finished);
}

void test_invalid_inputs_and_backend_configuration() {
  require_throws<std::invalid_argument>([] { Request({1}, -1, 0, 0); },
                                        "negative arrival was accepted");
  require_throws<std::invalid_argument>(
      [] { SimulatedBackend({-1, 0, 0, 0, 0}); },
      "negative backend cost was accepted");

  const auto backend = standard_backend();
  require_throws<std::invalid_argument>(
      [&] { backend.estimate_prefill_us(1, 0); },
      "zero active prefill batch was accepted");
  require_throws<std::invalid_argument>(
      [&] { backend.estimate_decode_step_us(0); },
      "zero active decode count was accepted");

  SimulationEngine engine(backend);
  engine.submit_request(Request({7}, 0, 1, 1));
  require_throws<std::invalid_argument>(
      [&] { engine.submit_request(Request({7}, 1, 1, 1)); },
      "duplicate request ID was accepted");
}

void require_empty_unchanged_engine(const SimulationEngine& engine) {
  require(!engine.failed(), "rejected submission failed the engine");
  require(engine.requests().empty(),
          "rejected submission changed the request map");
  require(engine.processed_event_count() == 0,
          "rejected submission changed the event log");
  require(engine.event_queue_empty(),
          "rejected submission created an event");
  require(engine.clock().now_us() == 0,
          "rejected submission changed the simulation clock");
}

void test_non_pristine_requests_are_rejected() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  const auto reject = [&](Request request, const std::string& message) {
    require_throws<std::invalid_argument>(
        [&] { engine.submit_request(std::move(request)); }, message);
    require_empty_unchanged_engine(engine);
  };

  Request prefilling({1}, 0, 1, 1);
  prefilling.transition_to(RequestState::Prefilling);
  reject(std::move(prefilling), "prefilling request was accepted");

  Request admitted({2}, 0, 1, 1);
  admitted.admitted_time_us = 0;
  reject(std::move(admitted), "request with admission time was accepted");

  Request generated({3}, 0, 1, 1);
  generated.generated_token_count = 1;
  reject(std::move(generated), "request with generated tokens was accepted");

  Request scheduled({4}, 0, 1, 1);
  scheduled.first_scheduled_time_us = 0;
  reject(std::move(scheduled), "scheduled request was accepted");

  Request first_token({5}, 0, 1, 1);
  first_token.first_token_time_us = 0;
  reject(std::move(first_token), "request with first-token time was accepted");

  Request finished({6}, 0, 1, 1);
  finished.finish_time_us = 0;
  reject(std::move(finished), "request with finish time was accepted");
}

void test_checked_arithmetic() {
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  const auto minimum = std::numeric_limits<std::int64_t>::min();
  require_throws<std::overflow_error>(
      [&] { (void)llm_lab::serving::checked_add(maximum, 1); },
      "addition overflow was not detected");
  require_throws<std::overflow_error>(
      [&] { (void)llm_lab::serving::checked_add(minimum, -1); },
      "negative addition overflow was not detected");
  require_throws<std::overflow_error>(
      [&] { (void)llm_lab::serving::checked_multiply(maximum, 2); },
      "multiplication overflow was not detected");
  require_throws<std::overflow_error>(
      [&] { (void)llm_lab::serving::checked_multiply(minimum, -1); },
      "minimum multiplication overflow was not detected");

  const SimulatedBackend overflowing({1, maximum, 0, 0, 0});
  require_throws<std::overflow_error>(
      [&] { (void)overflowing.estimate_prefill_us(2, 1); },
      "backend cost overflow was not detected");
}

void test_fcfs_and_later_arrival() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({20}, 0, 5, 2));
  engine.submit_request(Request({10}, 0, 1, 1));
  engine.submit_request(Request({30}, 4, 0, 1));
  engine.run();

  const auto& first = engine.request({20});
  const auto& second = engine.request({10});
  const auto& later = engine.request({30});
  require(first.admitted_time_us.has_value() &&
              first.finish_time_us.has_value() &&
              second.admitted_time_us.has_value() &&
              second.finish_time_us.has_value() &&
              later.admitted_time_us.has_value() &&
              later.finish_time_us.has_value(),
          "FIFO scenario has a missing lifecycle timestamp");
  require(*second.admitted_time_us == 0 && *second.finish_time_us == 18,
          "request-ID tie-break winner has unexpected timestamps");
  require(*first.admitted_time_us == 18 && *first.finish_time_us == 49,
          "second FCFS request has unexpected timestamps");
  require(*later.admitted_time_us == 49 && *later.finish_time_us == 65,
          "later FCFS request has unexpected timestamps");
  require(*first.admitted_time_us >= *second.finish_time_us,
          "second FCFS request was admitted before the first finished");
  require(*first.admitted_time_us < *later.admitted_time_us,
          "later arrival bypassed an earlier request");
  require(*later.admitted_time_us >= later.arrival_time_us,
          "request executed before arrival");
  require(*later.admitted_time_us == *first.finish_time_us,
          "later arrival did not wait behind active work");

  bool first_completed = false;
  for (const Event& event : engine.event_log()) {
    if (event.request_id != RequestId{10}) {
      continue;
    }
    require(!first_completed, "finished request was scheduled again");
    if (event.type == EventType::RequestComplete) {
      first_completed = true;
    }
  }
  require(first_completed, "first request had no completion event");
  require(engine.event_queue_empty(), "events remain after FCFS scenario");
}

void test_head_of_line_blocking_and_simultaneous_id_order() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({50}, 0, 5, 2));
  engine.submit_request(Request({20}, 1, 10, 1));
  engine.submit_request(Request({30}, 2, 0, 0));
  engine.run();

  const auto& long_waiter = engine.request({20});
  const auto& short_waiter = engine.request({30});
  require(long_waiter.admitted_time_us.has_value() &&
              long_waiter.finish_time_us.has_value() &&
              short_waiter.admitted_time_us.has_value(),
          "head-of-line scenario is missing timestamps");
  require(*long_waiter.admitted_time_us == 31 &&
              *long_waiter.finish_time_us == 67 &&
              *short_waiter.admitted_time_us == 67,
          "later short request bypassed earlier long request");

  SimulationEngine tied(backend);
  tied.submit_request(Request({9}, 0, 0, 0));
  tied.submit_request(Request({2}, 0, 0, 0));
  tied.run();
  const auto& lower_id = tied.request({2});
  const auto& higher_id = tied.request({9});
  require(lower_id.admitted_time_us.has_value() &&
              higher_id.admitted_time_us.has_value(),
          "simultaneous requests were not admitted");
  require(*lower_id.admitted_time_us == 0 &&
              *higher_id.admitted_time_us == 11,
          "simultaneous arrivals did not use request ID order");
}

void test_waiting_and_future_arrival_cancellation() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 4, 2));
  engine.submit_request(Request({2}, 0, 1, 1));
  require(engine.run_next_timestamp(), "arrival timestamp was not processed");
  require(engine.scheduler_waiting_count() == 1,
          "second request is not waiting");
  engine.cancel_request({2});
  engine.run();

  const auto& cancelled = engine.request({2});
  require(cancelled.state == RequestState::Cancelled,
          "waiting request did not become Cancelled");
  require(cancelled.finish_time_us.has_value() &&
              *cancelled.finish_time_us == 0,
          "waiting cancellation has wrong finish time");
  require(!cancelled.admitted_time_us.has_value(),
          "waiting cancelled request was admitted");
  require(cancelled.generated_token_count == 0,
          "waiting cancelled request generated tokens");

  SimulationEngine future(backend);
  future.submit_request(Request({8}, 5, 1, 1));
  future.cancel_request({8});
  future.run();
  const auto& future_cancelled = future.request({8});
  require(future_cancelled.state == RequestState::Cancelled &&
              future_cancelled.finish_time_us.has_value() &&
              *future_cancelled.finish_time_us == 0,
          "future-arrival cancellation has wrong terminal state");
  require(future.ignored_cancelled_event_count() == 1,
          "stale future arrival was not counted");
  require(future.event_log().empty(),
          "ignored future arrival entered the successful event log");
  const auto& statistics = future.scheduler_statistics();
  require(statistics.arrived_request_count == 0 &&
              statistics.cancelled_request_count == 1,
          "future cancellation scheduler statistics are wrong");
}

void test_active_cancellation_and_stale_events() {
  const auto backend = standard_backend();

  SimulationEngine prefill(backend);
  prefill.submit_request(Request({1}, 0, 2, 2));
  require(prefill.run_next_timestamp(), "prefill arrival was not processed");
  require(prefill.request({1}).state == RequestState::Prefilling,
          "request did not enter prefill");
  prefill.cancel_request({1});
  const std::size_t prefill_log_size = prefill.event_log().size();
  prefill.run();
  require(prefill.request({1}).generated_token_count == 0,
          "prefill-cancelled request generated a token");
  require(prefill.ignored_cancelled_event_count() == 1,
          "stale prefill completion was not counted");
  require(prefill.event_queue_empty(),
          "stale prefill completion did not drain");
  require(prefill.event_log().size() == prefill_log_size,
          "ignored prefill completion entered the successful event log");

  SimulationEngine decode(backend);
  decode.submit_request(Request({1}, 0, 1, 3));
  require(decode.run_next_timestamp(), "decode arrival was not processed");
  require(decode.run_next_timestamp(), "prefill completion was not processed");
  require(decode.request({1}).state == RequestState::Decoding,
          "request did not enter decode");
  require(decode.run_next_timestamp(), "first decode step was not processed");
  require(decode.request({1}).generated_token_count == 1,
          "first decode token was not generated");
  decode.cancel_request({1});
  const std::size_t decode_log_size = decode.event_log().size();
  decode.run();
  const auto& cancelled = decode.request({1});
  require(cancelled.state == RequestState::Cancelled &&
              cancelled.finish_time_us.has_value(),
          "decode cancellation is not terminal");
  require(cancelled.generated_token_count == 1,
          "decode-cancelled request produced an additional token");
  require(decode.ignored_cancelled_event_count() == 1,
          "stale decode event was not counted");
  require(decode.event_log().size() == decode_log_size,
          "ignored decode event entered the successful event log");
}

void test_cancelled_event_authorization_is_exact() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 2, 2));
  require(engine.run_next_timestamp(), "arrival timestamp was not processed");
  llm_lab::serving::test::SimulationEngineTestAccess::inject_event(
      engine, 15, {1}, EventType::PrefillComplete);
  engine.cancel_request({1});
  require_throws<std::logic_error>(
      [&] { engine.run(); },
      "duplicate event for cancelled request was silently ignored");
  require(engine.failed(), "duplicate cancelled event did not fail engine");
  require(engine.ignored_cancelled_event_count() == 1,
          "legitimate cancelled event was not ignored exactly once");
  require(engine.processed_event_count() == 1,
          "ignored or corrupt event entered successful event log");
}

void test_arbitrary_and_finished_stale_events_fail() {
  const auto backend = standard_backend();

  SimulationEngine arbitrary(backend);
  arbitrary.submit_request(Request({1}, 0, 1, 1));
  require(arbitrary.run_next_timestamp(), "arrival timestamp was not processed");
  llm_lab::serving::test::SimulationEngineTestAccess::inject_event(
      arbitrary, 0, {1}, EventType::RequestArrival);
  require_throws<std::logic_error>(
      [&] { arbitrary.run(); }, "arbitrary stale event did not fail engine");
  require(arbitrary.failed() &&
              arbitrary.ignored_cancelled_event_count() == 0,
          "arbitrary stale event used cancellation authorization");

  SimulationEngine finished(backend);
  finished.submit_request(Request({2}, 0, 0, 0));
  finished.run();
  const std::int64_t finish_time_us = finished.clock().now_us();
  llm_lab::serving::test::SimulationEngineTestAccess::inject_event(
      finished, finish_time_us, {2}, EventType::RequestComplete);
  require_throws<std::logic_error>(
      [&] { finished.run(); }, "finished-request stale event did not fail");
  require(finished.failed() && finished.ignored_cancelled_event_count() == 0,
          "finished stale event used cancellation authorization");
}

void test_cancellation_defers_for_current_timestamp_arrival() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 5, 2));
  engine.submit_request(Request({20}, 0, 0, 1));
  require(engine.run_next_timestamp(), "initial arrivals were not processed");
  require(engine.request({1}).state == RequestState::Prefilling &&
              engine.scheduler_waiting_count() == 1,
          "initial active/waiting state is wrong");

  engine.submit_request(Request({10}, 0, 0, 1));
  engine.cancel_request({1});
  require(!engine.request({20}).admitted_time_us.has_value(),
          "existing waiter was admitted before current-time arrival drained");
  require(engine.scheduler_running_count() == 0,
          "cancellation did not defer current-time admission");

  require(engine.run_next_timestamp(),
          "pending current-time arrival was not processed");
  const auto& lower_id = engine.request({10});
  const auto& higher_id = engine.request({20});
  require(lower_id.state == RequestState::Prefilling &&
              lower_id.admitted_time_us.has_value() &&
              *lower_id.admitted_time_us == 0,
          "lower-ID current-time arrival was not admitted first");
  require(!higher_id.admitted_time_us.has_value(),
          "higher-ID waiter bypassed current-time arrival");
  engine.run();
  require(engine.request({20}).state == RequestState::Finished,
          "deferred higher-ID waiter did not eventually finish");
}

void test_cancellation_admits_next_and_rejections() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 5, 2));
  engine.submit_request(Request({2}, 0, 0, 1));
  require(engine.run_next_timestamp(), "arrival timestamp was not processed");
  engine.cancel_request({1});
  const auto& next = engine.request({2});
  require(next.state == RequestState::Prefilling &&
              next.admitted_time_us.has_value() &&
              *next.admitted_time_us == 0,
          "active cancellation did not admit the next waiter");
  require_throws<std::logic_error>(
      [&] { engine.cancel_request({1}); },
      "repeated cancellation was accepted");
  require_throws<std::out_of_range>(
      [&] { engine.cancel_request({999}); },
      "unknown cancellation was accepted");
  engine.run();
  require(engine.request({2}).state == RequestState::Finished,
          "replacement request did not finish");
  require(engine.ignored_cancelled_event_count() == 1,
          "cancelled active request event was not consumed");
  require_throws<std::logic_error>(
      [&] { engine.cancel_request({2}); },
      "finished request cancellation was accepted");

  const auto& statistics = engine.scheduler_statistics();
  require(statistics.arrived_request_count == 2 &&
              statistics.admitted_request_count == 2 &&
              statistics.completed_request_count == 1 &&
              statistics.cancelled_request_count == 1,
          "active cancellation statistics are wrong");
}

void test_scheduler_statistics_exact_timestamps() {
  const auto backend = standard_backend();
  SimulationEngine engine(backend);
  engine.submit_request(Request({2}, 0, 0, 1));
  engine.submit_request(Request({1}, 0, 0, 1));
  engine.submit_request(Request({3}, 3, 0, 1));
  engine.run();

  const auto& first = engine.request({1});
  const auto& second = engine.request({2});
  const auto& third = engine.request({3});
  require(first.admitted_time_us.has_value() &&
              second.admitted_time_us.has_value() &&
              third.admitted_time_us.has_value(),
          "statistics scenario is missing admission times");
  require(*first.admitted_time_us == 0 && *second.admitted_time_us == 16 &&
              *third.admitted_time_us == 32,
          "statistics scenario admission timestamps are wrong");
  const auto& statistics = engine.scheduler_statistics();
  require(statistics.arrived_request_count == 3 &&
              statistics.admitted_request_count == 3 &&
              statistics.completed_request_count == 3 &&
              statistics.cancelled_request_count == 0,
          "scheduler cumulative counts are wrong");
  require(statistics.maximum_waiting_queue_depth == 2,
          "engine maximum queue depth is wrong");
  require(statistics.total_queue_wait_time_us == 45,
          "engine total queue wait is wrong");
  require(engine.scheduler_waiting_count() == 0 &&
              engine.scheduler_running_count() == 0,
          "engine current scheduler counts are not zero");
}

void test_engine_capacity_configuration() {
  const auto backend = standard_backend();
  require_throws<std::invalid_argument>(
      [&] { SimulationEngine invalid(backend, SimulationEngineConfig{0}); },
      "zero engine scheduler capacity was accepted");
  require_throws<std::invalid_argument>(
      [&] { SimulationEngine invalid(backend, SimulationEngineConfig{2}); },
      "unsupported multi-active engine capacity was accepted");
}

void test_timestamp_overflow() {
  const SimulatedBackend backend({1, 0, 0, 0, 0});
  SimulationEngine engine(backend);
  engine.submit_request(
      Request({1}, std::numeric_limits<std::int64_t>::max(), 0, 0));
  require_throws<std::overflow_error>([&] { engine.run(); },
                                      "timestamp overflow was not detected");
  require(engine.failed(), "timestamp overflow did not fail the engine");
  require(engine.processed_event_count() == 0,
          "throwing arrival event was recorded as processed");
  require_throws<std::logic_error>([&] { engine.run(); },
                                   "failed engine ran a second time");
  require_throws<std::logic_error>([&] { (void)engine.request({1}); },
                                   "failed request result remained available");
  require_throws<std::logic_error>([&] { (void)engine.event_log(); },
                                   "failed event log remained available");
  require_throws<std::logic_error>([&] { (void)engine.clock(); },
                                   "failed clock remained available");
}

void test_decode_backend_exception_is_terminal() {
  const ThrowingDecodeBackend backend;
  SimulationEngine engine(backend);
  engine.submit_request(Request({1}, 0, 1, 1));

  require_throws<std::runtime_error>([&] { engine.run(); },
                                     "decode backend exception was not propagated");
  require(engine.failed(), "decode exception did not fail the engine");
  require(engine.processed_event_count() == 1,
          "throwing prefill event was recorded as processed");
  require_throws<std::logic_error>([&] { engine.run(); },
                                   "decode-failed engine ran a second time");
  require_throws<std::logic_error>([&] { (void)engine.requests(); },
                                   "failed request map remained available");
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"single lifecycle and time zero", test_single_request_lifecycle_and_time_zero},
      {"zero prompt and zero output", test_zero_prompt_and_zero_output},
      {"first token once", test_first_token_assigned_once},
      {"stable event ordering", test_event_ordering_is_stable},
      {"monotonic clock", test_clock_is_monotonic},
      {"deterministic repetition", test_repeated_execution_is_deterministic},
      {"state transitions", test_state_transitions},
      {"invalid inputs", test_invalid_inputs_and_backend_configuration},
      {"non-pristine requests", test_non_pristine_requests_are_rejected},
      {"checked arithmetic", test_checked_arithmetic},
      {"FCFS and later arrival", test_fcfs_and_later_arrival},
      {"head-of-line and ID order",
       test_head_of_line_blocking_and_simultaneous_id_order},
      {"waiting and future cancellation",
       test_waiting_and_future_arrival_cancellation},
      {"active cancellation and stale events",
       test_active_cancellation_and_stale_events},
      {"exact cancelled-event authorization",
       test_cancelled_event_authorization_is_exact},
      {"arbitrary stale events", test_arbitrary_and_finished_stale_events_fail},
      {"current-time cancellation admission",
       test_cancellation_defers_for_current_timestamp_arrival},
      {"cancellation admits next", test_cancellation_admits_next_and_rejections},
      {"scheduler statistics", test_scheduler_statistics_exact_timestamps},
      {"engine capacity", test_engine_capacity_configuration},
      {"timestamp overflow", test_timestamp_overflow},
      {"decode backend failure", test_decode_backend_exception_is_terminal},
  };

  std::size_t failures = 0;
  for (const auto& test : tests) {
    try {
      test.second();
      std::cout << "PASS: " << test.first << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  return 0;
}
