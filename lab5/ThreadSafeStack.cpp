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
#include <vector>

template <typename T>
class ThreadSafeStack {
    private:
        std::stack<T> stack_;
        std::condition_variable condVar_;
        std::mutex writeMutex_;
        std::mutex topMutex_;
        std::shared_timed_mutex readMutex_;

    public:
        ThreadSafeStack() {}

        void push(T item) {
            std::unique_lock<std::mutex> writerLock(writeMutex_);
            stack_.push(item);
            //notify the thread which is waiting
            condVar_.notify_all();
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

        std::optional<T> top_with_waiting() {
            // requires shared ownership to read from other
            if (!stack_.empty()) {
                auto item = this->top();
                return item;
            }
        
            std::unique_lock<std::mutex> readerLock(topMutex_);
            condVar_.wait(readerLock, [this]() { return !stack_.empty(); });
            auto item = this->top();
            return item;
        }

        std::optional<T> top() {
            // requires shared ownership to read from other
            std::shared_lock<std::shared_timed_mutex> readerLock(readMutex_);
            if (stack_.empty()) {
                return std::nullopt;
            }
            auto item = stack_.top();
            return item;
        }
};

/*int main() {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();
    std::vector<std::thread> threads;

    for (int i = 0; i < 1; i++) {
        threads.push_back(std::thread( [stack]() {
            if (5 != stack->top_with_waiting().value_or(-1)) {
                fprintf(stderr, "false\n");
            }
        }));
    }
    std::thread t1 = std::thread([stack]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        stack->push(5);
    });
    for (auto &t : threads) {
        t.join();
    }
    t1.join();

}*/