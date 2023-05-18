#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ThreadSafeStack.cpp"

//g++ main.cpp -o test -lgtest -lgmock -pthread

TEST(ThreadSafeStackTest, ThreadSafeStackTest_Push_Pop) {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();

    stack->push(1);
    stack->push(2);

    EXPECT_EQ(2, stack->pop_without_waiting().value_or(-1));
    EXPECT_EQ(1, stack->pop_without_waiting().value_or(-1));
    EXPECT_EQ(-1, stack->pop_without_waiting().value_or(-1));
}

TEST(ThreadSafeStackTest, ThreadSafeStackTest_Top) {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();

    stack->push(1);
    stack->push(2);

    EXPECT_EQ(2, stack->top().value_or(-1));
    EXPECT_EQ(2, stack->top().value_or(-1));
    EXPECT_EQ(2, stack->pop_without_waiting().value_or(-1));
}

TEST(ThreadSafeStackTest, ThreadSafeStackTest_ConcurrentPushAndPop) {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();
    std::vector<std::thread> threads;

    for (int i = 0; i < 1000; i++) {
        threads.emplace_back( [stack, i]() {
            stack->push(i);
            stack->pop_without_waiting();
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    EXPECT_EQ(-1, stack->pop_without_waiting().value_or(-1));
}

TEST(ThreadSafeStackTest, ThreadSafeStackTest_ConcurrentTop) {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();
    stack->push(1);
    stack->push(2);

    std::vector<std::thread> threads;

    for (int i = 0; i < 1000; i++) {
        threads.emplace_back( [stack]() {
            EXPECT_TRUE(stack->top().value_or(-1) == 2);
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    EXPECT_EQ(2, stack->pop_with_waiting());
    EXPECT_EQ(1, stack->pop_with_waiting());
    EXPECT_EQ(-1, stack->pop_without_waiting().value_or(-1));
}

TEST(ThreadSafeStackTest, ThreadSafeStackTest_Pop) {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();
    int value = 0;

    std::thread t1( [stack, &value]() {
        stack->top();
        value = stack->pop_with_waiting();	
    });

    std::thread t2( [stack]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
	    stack->push(1);	    
    });
    t1.join();
    t2.join();
    EXPECT_EQ(value, 1);
    EXPECT_EQ(-1, stack->pop_without_waiting().value_or(-1));
}

TEST(ThreadSafeStackTest, MoreConcurrentPushAndPop) {
    ThreadSafeStack<int> *stack = new ThreadSafeStack<int>();
    std::vector<std::thread> threads;

    for (int i = 0; i < 100; i++) {
        threads.push_back(std::thread( [stack, i]() {
            for (int j = 0; j < 100; j++) {
                stack->push(i * 100 + j);
            }
        }));
    }

    for (auto &t : threads) {
        t.join();
    }

    std::vector<std::optional<int>> results(100 * 100);
    std::vector<std::thread> popThreads;

    for (int i = 0; i < 100; i++) {
        popThreads.push_back(std::thread( [stack, &results, i]() {
            for (int j = 0; j < 100; j++) {
                results[i * 100 + j] = stack->pop_without_waiting();
            }
        }));
    }

    for (auto &t : popThreads) {
        t.join();
    }

    std::sort(results.begin(), results.end());

    for (int i = 0; i < 100 * 100; i++) {
        EXPECT_EQ(results[i], i);
    }

    EXPECT_EQ(-1, stack->pop_without_waiting().value_or(-1));
}

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}