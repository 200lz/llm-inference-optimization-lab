#include "serving/fcfs_scheduler.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using llm_lab::serving::FcfsScheduler;
using llm_lab::serving::RequestId;
using llm_lab::serving::SchedulerDecision;
using llm_lab::serving::SchedulerDecisionKind;

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

RequestId admitted_id(const SchedulerDecision& decision) {
  require(decision.kind() == SchedulerDecisionKind::Admit,
          "scheduler did not return Admit");
  require(decision.request_id().has_value(), "Admit has no request ID");
  return *decision.request_id();
}

void admit_next(FcfsScheduler& scheduler, std::int64_t timestamp_us) {
  const SchedulerDecision decision = scheduler.choose_next_request();
  require(decision.kind() == SchedulerDecisionKind::Admit,
          "expected an admission decision");
  scheduler.notify_admitted(decision, timestamp_us);
}

static_assert(!std::is_default_constructible_v<SchedulerDecision>,
              "SchedulerDecision must not have an invalid default state");
static_assert(!std::is_aggregate_v<SchedulerDecision>,
              "SchedulerDecision must preserve its representation invariant");

void test_empty_and_invalid_capacity() {
  FcfsScheduler scheduler;
  const auto decision = scheduler.choose_next_request();
  require(decision.kind() == SchedulerDecisionKind::NoWork,
          "empty scheduler did not return NoWork");
  require(!decision.request_id().has_value(), "NoWork contains a request ID");
  require(scheduler.empty(), "new scheduler is not empty");
  require_throws<std::invalid_argument>([] { FcfsScheduler invalid(0); },
                                        "zero capacity was accepted");
}

void test_fcfs_order_and_tie_break() {
  FcfsScheduler scheduler;
  scheduler.notify_arrival({9}, 5);
  scheduler.notify_arrival({7}, 1);
  scheduler.notify_arrival({3}, 5);

  require(admitted_id(scheduler.choose_next_request()) == RequestId{7},
          "earliest arrival was not selected");
  admit_next(scheduler, 6);
  require(scheduler.choose_next_request().kind() ==
              SchedulerDecisionKind::CapacityFull,
          "running scheduler did not report CapacityFull");
  scheduler.notify_completed({7}, 8);
  require(admitted_id(scheduler.choose_next_request()) == RequestId{3},
          "request ID did not break simultaneous-arrival tie");
  admit_next(scheduler, 8);
  scheduler.notify_completed({3}, 9);
  require(admitted_id(scheduler.choose_next_request()) == RequestId{9},
          "remaining FCFS request was not selected");
}

void test_duplicate_and_terminal_validation() {
  FcfsScheduler scheduler;
  scheduler.notify_arrival({1}, 0);
  require_throws<std::logic_error>(
      [&] { scheduler.notify_arrival({1}, 0); },
      "duplicate arrival was accepted");
  const SchedulerDecision admission = scheduler.choose_next_request();
  scheduler.notify_admitted(admission, 0);
  require_throws<std::logic_error>(
      [&] { scheduler.notify_admitted(admission, 0); },
      "duplicate admission was accepted");
  scheduler.notify_completed({1}, 1);
  require(scheduler.empty(), "completed request remained in scheduler");
  require_throws<std::logic_error>(
      [&] { scheduler.notify_completed({1}, 1); },
      "duplicate completion was accepted");
}

void test_cancellation_removes_waiting_and_running() {
  FcfsScheduler scheduler;
  scheduler.notify_arrival({1}, 0);
  scheduler.notify_arrival({2}, 0);
  scheduler.notify_cancelled({1}, 1);
  require(scheduler.waiting_count() == 1,
          "waiting cancellation did not remove request");
  require(admitted_id(scheduler.choose_next_request()) == RequestId{2},
          "cancelled waiting request was selected");
  admit_next(scheduler, 2);
  scheduler.notify_cancelled({2}, 3);
  require(scheduler.empty(), "active cancellation left scheduler state");
}

void test_statistics_and_queue_depth() {
  FcfsScheduler scheduler;
  scheduler.notify_arrival({3}, 2);
  scheduler.notify_arrival({1}, 2);
  scheduler.notify_arrival({8}, 4);
  admit_next(scheduler, 5);  // request 1, wait 3
  scheduler.notify_completed({1}, 7);
  scheduler.notify_cancelled({3}, 7);  // no admitted wait contribution
  admit_next(scheduler, 10);  // request 8, wait 6
  scheduler.notify_cancelled({8}, 12);

  const auto& statistics = scheduler.statistics();
  require(statistics.arrived_request_count == 3, "wrong arrived count");
  require(statistics.admitted_request_count == 2, "wrong admitted count");
  require(statistics.completed_request_count == 1, "wrong completed count");
  require(statistics.cancelled_request_count == 2, "wrong cancelled count");
  require(statistics.maximum_waiting_queue_depth == 3,
          "wrong maximum queue depth");
  require(statistics.total_queue_wait_time_us == 9,
          "wrong total queue wait");
  require(scheduler.waiting_count() == 0 && scheduler.running_count() == 0,
          "current scheduler counts are wrong");
}

void test_queue_wait_overflow() {
  FcfsScheduler scheduler;
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  scheduler.notify_arrival({1}, 0);
  scheduler.notify_arrival({2}, 0);
  admit_next(scheduler, maximum);
  scheduler.notify_completed({1}, maximum);
  require_throws<std::overflow_error>(
      [&] {
        const auto decision = scheduler.choose_next_request();
        scheduler.notify_admitted(decision, maximum);
      },
      "total queue wait overflow was not detected");
  require(scheduler.waiting_count() == 1 && scheduler.running_count() == 0,
          "overflowing admission mutated scheduler state");
}

void test_decision_representation_and_no_work_semantics() {
  const SchedulerDecision admit = SchedulerDecision::admit({4}, 7);
  require(admit.kind() == SchedulerDecisionKind::Admit &&
              admit.request_id().has_value() &&
              *admit.request_id() == RequestId{4} &&
              admit.decision_epoch() == 7,
          "valid Admit decision lost its payload");
  const SchedulerDecision no_work = SchedulerDecision::no_work(8);
  require(no_work.kind() == SchedulerDecisionKind::NoWork &&
              !no_work.request_id().has_value() &&
              no_work.decision_epoch() == 8,
          "valid NoWork decision has an invalid payload");
  const SchedulerDecision full = SchedulerDecision::capacity_full(9);
  require(full.kind() == SchedulerDecisionKind::CapacityFull &&
              !full.request_id().has_value() &&
              full.decision_epoch() == 9,
          "valid CapacityFull decision has an invalid payload");

  FcfsScheduler scheduler;
  scheduler.notify_arrival({1}, 0);
  admit_next(scheduler, 0);
  require(scheduler.choose_next_request().kind() ==
              SchedulerDecisionKind::NoWork,
          "running request without a waiter did not return NoWork");
  scheduler.notify_arrival({2}, 1);
  require(scheduler.choose_next_request().kind() ==
              SchedulerDecisionKind::CapacityFull,
          "waiting request at full capacity did not return CapacityFull");
}

void test_fcfs_commit_and_stale_decisions() {
  FcfsScheduler scheduler;
  scheduler.notify_arrival({10}, 0);
  scheduler.notify_arrival({20}, 1);
  const SchedulerDecision head = scheduler.choose_next_request();
  const auto before = scheduler.statistics();
  const SchedulerDecision bypass =
      SchedulerDecision::admit({20}, head.decision_epoch());
  require_throws<std::logic_error>(
      [&] { scheduler.notify_admitted(bypass, 2); },
      "non-head FCFS admission was accepted");
  require(scheduler.waiting_count() == 2 && scheduler.running_count() == 0,
          "rejected non-head admission mutated scheduler state");
  require(scheduler.statistics().admitted_request_count ==
              before.admitted_request_count &&
              scheduler.statistics().total_queue_wait_time_us ==
                  before.total_queue_wait_time_us &&
              admitted_id(scheduler.choose_next_request()) == RequestId{10},
          "rejected non-head admission mutated order or statistics");

  scheduler.notify_cancelled({10}, 2);
  require_throws<std::logic_error>(
      [&] { scheduler.notify_admitted(head, 2); },
      "decision survived a scheduler state change");
  const SchedulerDecision fresh = scheduler.choose_next_request();
  require(admitted_id(fresh) == RequestId{20},
          "fresh decision did not select the new head");
  scheduler.notify_admitted(fresh, 2);

  FcfsScheduler inserted;
  inserted.notify_arrival({20}, 5);
  const SchedulerDecision old = inserted.choose_next_request();
  inserted.notify_arrival({10}, 5);
  require_throws<std::logic_error>(
      [&] { inserted.notify_admitted(old, 5); },
      "decision survived insertion of a new FCFS head");
  const SchedulerDecision replacement = inserted.choose_next_request();
  require(admitted_id(replacement) == RequestId{10},
          "fresh decision ignored newly inserted FCFS head");
  inserted.notify_admitted(replacement, 5);
}

void test_duplicate_cancellation_notifications() {
  FcfsScheduler waiting;
  waiting.notify_arrival({1}, 0);
  waiting.notify_cancelled({1}, 1);
  const auto waiting_stats = waiting.statistics();
  require_throws<std::logic_error>(
      [&] { waiting.notify_cancelled({1}, 1); },
      "duplicate waiting cancellation was accepted");
  require(waiting.statistics().cancelled_request_count ==
              waiting_stats.cancelled_request_count &&
              waiting.waiting_count() == 0 && waiting.running_count() == 0,
          "duplicate waiting cancellation mutated scheduler state");

  FcfsScheduler running;
  running.notify_arrival({2}, 0);
  admit_next(running, 0);
  running.notify_cancelled({2}, 1);
  const auto running_stats = running.statistics();
  require_throws<std::logic_error>(
      [&] { running.notify_cancelled({2}, 1); },
      "duplicate running cancellation was accepted");
  require(running.statistics().cancelled_request_count ==
              running_stats.cancelled_request_count &&
              running.running_count() == 0,
          "duplicate running cancellation mutated scheduler state");

  FcfsScheduler prearrival;
  prearrival.notify_cancelled({3}, 0);
  const auto prearrival_stats = prearrival.statistics();
  require_throws<std::logic_error>(
      [&] { prearrival.notify_cancelled({3}, 0); },
      "duplicate pre-arrival cancellation was accepted");
  require(prearrival.statistics().cancelled_request_count ==
              prearrival_stats.cancelled_request_count &&
              prearrival.empty(),
          "duplicate pre-arrival cancellation mutated scheduler state");
}

void test_capacity_two() {
  FcfsScheduler scheduler(2);
  scheduler.notify_arrival({30}, 1);
  scheduler.notify_arrival({10}, 0);
  scheduler.notify_arrival({20}, 1);
  require(admitted_id(scheduler.choose_next_request()) == RequestId{10},
          "capacity-two scheduler selected wrong first request");
  admit_next(scheduler, 2);
  require(admitted_id(scheduler.choose_next_request()) == RequestId{20},
          "capacity-two scheduler selected wrong second request");
  admit_next(scheduler, 3);
  require(scheduler.running_count() == 2 && scheduler.waiting_count() == 1,
          "capacity-two scheduler has wrong current counts");
  require(scheduler.choose_next_request().kind() ==
              SchedulerDecisionKind::CapacityFull,
          "capacity-two scheduler exceeded its limit");
  scheduler.notify_completed({10}, 4);
  require(admitted_id(scheduler.choose_next_request()) == RequestId{30},
          "freed capacity did not expose next FCFS request");
  admit_next(scheduler, 5);
  const auto& statistics = scheduler.statistics();
  require(statistics.arrived_request_count == 3 &&
              statistics.admitted_request_count == 3 &&
              statistics.completed_request_count == 1 &&
              statistics.total_queue_wait_time_us == 8,
          "capacity-two scheduler statistics are wrong");
}

void test_prearrival_cancellation_accounting() {
  FcfsScheduler scheduler;
  scheduler.notify_cancelled({11}, 0);
  const auto& statistics = scheduler.statistics();
  require(statistics.arrived_request_count == 0,
          "pre-arrival cancellation counted as arrival");
  require(statistics.cancelled_request_count == 1,
          "pre-arrival cancellation was not counted");
  require(scheduler.empty(), "pre-arrival cancellation created work");
  require_throws<std::logic_error>(
      [&] { scheduler.notify_arrival({11}, 5); },
      "cancelled request was allowed to arrive");
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"empty and invalid capacity", test_empty_and_invalid_capacity},
      {"FCFS order and tie break", test_fcfs_order_and_tie_break},
      {"duplicate and terminal validation",
       test_duplicate_and_terminal_validation},
      {"cancellation removal", test_cancellation_removes_waiting_and_running},
      {"statistics and queue depth", test_statistics_and_queue_depth},
      {"queue wait overflow", test_queue_wait_overflow},
      {"pre-arrival cancellation", test_prearrival_cancellation_accounting},
      {"decision representation", test_decision_representation_and_no_work_semantics},
      {"FCFS commit and stale decisions", test_fcfs_commit_and_stale_decisions},
      {"duplicate cancellations", test_duplicate_cancellation_notifications},
      {"capacity two", test_capacity_two},
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
