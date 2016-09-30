#pragma once

namespace StreamCompaction {
namespace Efficient {

    //StreamCompaction::Common::PerformanceTimer& timer();

    void scan(int n, int *odata, const int *idata);

    int compact(int n, int *odata, const int *idata);

    int getUpSweepBlockSize();
    int getDownSweepBlockSize();

    void scanInPlaceDevice(int extended_n, int* dev_buffer);
}
}
