// Which cached things to drop to stay under a memory budget.
//
// Every instance used to keep its analysis -- a label image per grouping mode
// on the host, its copy on the GPU, and two blur tiles -- for as long as the
// host kept the instance. Resolve keeps one per use of the effect for the whole
// session, so a long timeline full of titles held all of it at once, and a
// render slowed as it went because nothing it had finished with was given
// back. What is in use at any moment is only what renders at that moment.
//
// The policy: over budget, drop the longest-unused first, but never anything
// used in the last few seconds. That window is what keeps several titles on
// screen together from evicting each other on every frame -- the budget may be
// exceeded rather than thrash. Host-independent and lock-free; callers lock.
#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>

namespace rta {

class LruBudget {
 public:
  // Record a use of `key` at time `now` (seconds, any monotonic origin), and
  // what it currently holds.
  void touch(const void* key, size_t bytes, double now) {
    Entry& e = entries_[key];
    e.bytes = bytes;
    e.lastUse = now;
  }

  void remove(const void* key) { entries_.erase(key); }

  size_t total() const {
    size_t t = 0;
    for (const auto& kv : entries_) t += kv.second.bytes;
    return t;
  }

  // Keys to drop, oldest first, until the total would fit in `budget`. Skips
  // `keep` and anything used within `protect` seconds of `now`.
  std::vector<const void*> victims(size_t budget, double now, double protect,
                                   const void* keep = nullptr) const {
    size_t t = total();
    std::vector<const void*> out;
    if (t <= budget) return out;
    std::vector<std::pair<double, const void*>> byAge;
    for (const auto& kv : entries_)
      if (kv.first != keep && now - kv.second.lastUse >= protect)
        byAge.push_back({kv.second.lastUse, kv.first});
    std::sort(byAge.begin(), byAge.end());
    for (const auto& a : byAge) {
      if (t <= budget) break;
      out.push_back(a.second);
      t -= entries_.at(a.second).bytes;
    }
    return out;
  }

 private:
  struct Entry {
    size_t bytes = 0;
    double lastUse = 0.0;
  };
  std::map<const void*, Entry> entries_;
};

}  // namespace rta
