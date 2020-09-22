#pragma once

#include <stlfwd>

struct TDefaultLFCounter;

template <class T, class TCounter = TDefaultLFCounter, int memMode = 1>
class TLockFreeQueue;

template <class T, class TCounter = TDefaultLFCounter>
class TAutoLockFreeQueue;

template <class T, class TCounter = TDefaultLFCounter>
class TFastLockFreeQueue;

template <class T>
class TLockFreeStack;

class IThreadFactory;

struct IObjectInQueue;
class TThreadFactoryHolder;

using TThreadFunction = std::function<void()>;

class IThreadPool;
class TFakeThreadPool;
class TThreadPool;
class TAdaptiveThreadPool;
class TSimpleThreadPool;

template <class TQueueType, class TSlave>
class TThreadPoolBinder;
