#include <chrono>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

int main() 
{
    auto start = std::chrono::steady_clock::now();
    
    std::this_thread::sleep_for(10ms);
    std::cout << "Elapsed(ms)=" << since(start).count() << std::endl;
    
    start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(1ms);
    std::cout << "Elapsed(us)=" 
        << since<std::chrono::microseconds>(start).count() << std::endl;
}

