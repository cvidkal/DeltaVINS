#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace DeltaVins {

class AbstractModule {
   public:
    AbstractModule() { keep_running_.store(true); }

    virtual ~AbstractModule() {
        Stop();
        delete modules_thread_;
    }

    void Join() {
        if (Joinable()) _join();
    }
    void Start() {
        if (!run_) _start();
    }

    virtual void Stop() { this->_stop(); }
    void detach() {
        if (modules_thread_ != nullptr && run_)
            if (!detached_) {
                _detach();
            }
    }

    bool Joinable() const {
        if (modules_thread_ != nullptr) return modules_thread_->joinable();
        return false;
    }

    void WakeUpMovers() {
        std::lock_guard<std::mutex> lk(wake_up_mutex_);
        wake_up_condition_variable_.notify_one();
    }

    void WakeUpAndWait() {
        std::unique_lock<std::mutex> lck(serial_mutex_);
        {
            std::lock_guard<std::mutex> lk(wake_up_mutex_);
            wake_up_condition_variable_.notify_one();
        }
        serial_condition_variable_.wait(lck);
    }

    virtual void RunThread() {
        while (keep_running_) {
            // check for wake up request
            {
                std::unique_lock<std::mutex> ul(wake_up_mutex_);
                wake_up_condition_variable_.wait(ul, [this]() {
                    return this->HaveThingsTodo() || !this->keep_running_;
                });
            }
            // do something when wake up
            while (HaveThingsTodo() && this->keep_running_) {
                DoWhatYouNeedToDo();
            }
        }
    }

   protected:
    std::thread* modules_thread_ = nullptr;
    std::mutex wake_up_mutex_;
    std::mutex serial_mutex_;
    std::condition_variable wake_up_condition_variable_;
    std::condition_variable serial_condition_variable_;
    std::atomic_bool keep_running_;
    bool run_ = false;
    bool detached_ = false;

    virtual bool HaveThingsTodo() = 0;
    virtual void DoWhatYouNeedToDo() = 0;

    void WaitForThingsToBeDone() {
        std::unique_lock<std::mutex> lck(serial_mutex_);
        serial_condition_variable_.wait(lck);
    }

    void TellOthersThingsToBeDone() {
        std::unique_lock<std::mutex> ul(serial_mutex_);
        serial_condition_variable_.notify_all();
    }

   private:
    void _stop() {
        {
            keep_running_.store(false);
            std::lock_guard<std::mutex> lk(wake_up_mutex_);
            wake_up_condition_variable_.notify_one();
        }
        Join();
    }

    void _detach() {
        modules_thread_->detach();
        detached_ = true;
    }

    void _start() {
        if (run_) return;
        keep_running_.store(true);

        modules_thread_ = new std::thread([&]() { this->RunThread(); });
        run_ = true;
    }

    void _join() {
        modules_thread_->join();
        run_ = false;
    }
};
}  // namespace DeltaVins