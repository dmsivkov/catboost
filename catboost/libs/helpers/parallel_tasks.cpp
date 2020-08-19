#include "parallel_tasks.h"

#include <util/generic/cast.h>

template <typename LocalExecutorType>
void NCB::ExecuteTasksInParallel(TVector<std::function<void()>>* tasks, LocalExecutorType* localExecutor) {
    localExecutor->ExecRangeWithThrow(
        [&tasks](int id) {
            (*tasks)[id]();
            (*tasks)[id] = nullptr; // destroy early, do not wait for all tasks to finish
        },
        0,
        SafeIntegerCast<int>(tasks->size()),
        LocalExecutorType::WAIT_COMPLETE
    );
}
template
void NCB::ExecuteTasksInParallel<NPar::TLocalExecutor>(TVector<std::function<void()>>* tasks, NPar::TLocalExecutor* localExecutor);
template
void NCB::ExecuteTasksInParallel<OMPNPar::TLocalExecutor>(TVector<std::function<void()>>* tasks, OMPNPar::TLocalExecutor* localExecutor);
