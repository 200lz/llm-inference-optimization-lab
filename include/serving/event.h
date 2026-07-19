#pragma once

#include "serving/request.h"

#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace llm_lab::serving {

enum class EventType {
  RequestArrival,
  PrefillComplete,
  DecodeStepComplete,
  RequestComplete,
};

struct Event {
  std::int64_t timestamp_us;
  std::uint64_t event_sequence;
  RequestId request_id;
  EventType type;
};

class EventQueue {
 public:
  std::uint64_t schedule(std::int64_t timestamp_us, RequestId request_id,
                         EventType type) {
    if (timestamp_us < 0) {
      throw std::invalid_argument("event timestamp must be non-negative");
    }
    if (next_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("event sequence overflow");
    }
    const std::uint64_t event_sequence = next_sequence_;
    events_.push(Event{timestamp_us, event_sequence, request_id, type});
    ++next_sequence_;
    return event_sequence;
  }

  Event pop() {
    if (events_.empty()) {
      throw std::logic_error("cannot pop an empty event queue");
    }
    const Event event = events_.top();
    events_.pop();
    return event;
  }

  const Event& top() const {
    if (events_.empty()) {
      throw std::logic_error("cannot inspect an empty event queue");
    }
    return events_.top();
  }

  bool empty() const noexcept { return events_.empty(); }
  std::size_t size() const noexcept { return events_.size(); }

 private:
  struct LaterEvent {
    bool operator()(const Event& lhs, const Event& rhs) const noexcept {
      if (lhs.timestamp_us != rhs.timestamp_us) {
        return lhs.timestamp_us > rhs.timestamp_us;
      }
      return lhs.event_sequence > rhs.event_sequence;
    }
  };

  std::priority_queue<Event, std::vector<Event>, LaterEvent> events_;
  std::uint64_t next_sequence_{0};
};

}  // namespace llm_lab::serving
