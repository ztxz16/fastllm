#include "devices/disk/disk-moe-replacement-cost.h"
#include <stdexcept>
#define Require(...) do { if (!(__VA_ARGS__)) throw std::runtime_error(#__VA_ARGS__); } while (0)
#include <iostream>
using Cost=fastllm::DiskMoeReplacementCost;
int main() {
    // One extra expected CPU hit cannot pay for asymmetric H2D + D2H.
    Require(!Cost::Pays({100,88,false},{{100,80,true}},1,.4,.8));
    Require(Cost::Pays({100,96,false},{{100,80,true}},1,.4,.8));
    // A CPU backing copy removes D2H cost, rather than charging it twice.
    Require(Cost::Pays({100,88,false},{{100,80,false}},1,.4,.8));
    // A larger candidate may displace multiple records. Count all of them.
    Require(Cost::Pays({200,88,false},{{100,80,true}},1,.4,.8));
    Require(!Cost::Pays({200,88,false},{{100,80,true},{100,80,true}},1,.4,.8));
    // Cost is in microseconds for actual byte counts, not an unscaled ratio.
    Require(!Cost::Pays({100,96,false},{{200,80,true}},1,.4,.8));
    std::cout<<"PASS: disk replacement bytes, all victims, H2D and measured D2H"<<std::endl;
}
