#include <iostream>
#include <stack>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <optional>
#include <thread>
#include <chrono>
#include <thread>
#include <string>

template <typename T>
class ThreadSafeStack {
    private:
        std::stack<T> stack_;
        std::condition_variable condVar_;
        std::mutex writeMutex_;
        std::shared_timed_mutex readMutex_;

    public:
        ThreadSafeStack() {}

        void push(T item) {
            std::unique_lock<std::mutex> writerLock(writeMutex_);
            stack_.push(item);
            //notify the thread which is waiting
            condVar_.notify_one();
        }

        T pop_with_waiting() {
            std::unique_lock<std::mutex> writerLock(writeMutex_);
            //wait until stack is not empty
            //[this]() {return !(stack_.empty()) -- predicate
            condVar_.wait(writerLock, [this]() { return !stack_.empty(); } );
            T item = stack_.top();
            stack_.pop();
            return item;
        }

        std::optional<T> pop_without_waiting() {
            std::unique_lock<std::mutex> writerLock(writeMutex_);
            //wait until stack is not empty
            if (stack_.empty()) {
                return std::nullopt;
            }
            T item = stack_.top();
            stack_.pop();
            return item;
        }

        std::optional<T> top() {
            // requires shared ownership to read from other
            std::shared_lock<std::shared_timed_mutex> readerLock(readMutex_);
            if (stack_.empty()) {
                return std::nullopt;
            }
            T item = stack_.top();
            return item;
        }
};