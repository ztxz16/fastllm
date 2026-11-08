#include "utils/disk_prefix_format_v4.h"
#include "json11.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
using Json=json11::Json;
namespace fs=std::filesystem;
static void Require(bool ok,const std::string&s){if(!ok)throw std::runtime_error(s);}
struct Fault{size_t cap=0;int error=0;bool fired=false;int openError=0;size_t directCalls=0;};static Fault fault;
struct IoTrace{size_t directOpens=0,bufferedOpens=0,directReads=0,bufferedReads=0,alignmentErrors=0;};static IoTrace trace;
extern "C" int __real_open(const char*,int,...);
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __real_preadv(int,const iovec*,int,off_t);
extern "C" int __wrap_open(const char*p,int flags,...){
 if(flags&O_DIRECT){++trace.directOpens;if(fault.openError&&!fault.fired){fault.fired=true;errno=fault.openError;return -1;}}
 else ++trace.bufferedOpens;
 return __real_open(p,flags,0600);
}
static bool Inject(size_t n,bool direct){
 bool eligible=direct?++fault.directCalls>1:n>128;
 if(fault.error&&!fault.fired&&eligible){fault.fired=true;errno=fault.error;return true;}return false;
}
extern "C" ssize_t __wrap_pread(int fd,void*out,size_t n,off_t at){
 bool direct=(fcntl(fd,F_GETFL)&O_DIRECT)!=0;
 if(direct){++trace.directReads;if(reinterpret_cast<uintptr_t>(out)%4096||n%4096||at%4096){++trace.alignmentErrors;errno=EINVAL;return -1;}}
 else ++trace.bufferedReads;
 if(Inject(n,direct))return -1;
 size_t cap=fault.cap;if(direct&&cap)cap=std::max(size_t(4096),cap/4096*4096);
 return __real_pread(fd,out,cap?std::min(n,cap):n,at);
}
extern "C" ssize_t __wrap_preadv(int fd,const iovec*v,int count,off_t at){
 bool direct=(fcntl(fd,F_GETFL)&O_DIRECT)!=0;if(direct){++trace.directReads;++trace.alignmentErrors;errno=EINVAL;return -1;}++trace.bufferedReads;
 size_t total=0;for(int i=0;i<count;++i)total+=v[i].iov_len;if(Inject(total,false))return -1;
 if(!fault.cap)return __real_preadv(fd,v,count,at);
 iovec limited[3];int n=0;size_t left=fault.cap;
 for(int i=0;i<count&&left;++i){limited[n]=v[i];limited[n].iov_len=std::min(left,v[i].iov_len);left-=limited[n].iov_len;++n;}
 return __real_preadv(fd,limited,n,at);
}
namespace Baseline{namespace prefix_v4=fastllm::prefix_v4;using namespace prefix_v4;
static void Check(bool ok,const std::string&s){Require(ok,s);}
#include "record_baseline.inc"
}
namespace Buffered{namespace prefix_v4=fastllm::prefix_v4;using namespace prefix_v4;
static void Check(bool ok,const std::string&s){Require(ok,s);}
#include "record_direct.inc"
}
namespace Direct{namespace prefix_v4=fastllm::prefix_v4;using namespace prefix_v4;
static void Check(bool ok,const std::string&s){Require(ok,s);}
#include "record_direct.inc"
}
static std::string Bytes(size_t n){std::string s(n,'\0');for(size_t i=0;i<n;++i)s[i]=char((i*31+7)%251);return s;}
static Json Config(const char*p){std::ifstream f(p);Require(bool(f),"fixture_open");std::string text((std::istreambuf_iterator<char>(f)),{}),error;auto j=Json::parse(text,error);Require(error.empty(),"fixture_json");return j;}
static size_t Fds(){size_t n=0;for(const auto&p:fs::directory_iterator("/proc/self/fd")){(void)p;++n;}return n;}
using ReadFn=bool(*)(const fs::path&,uint64_t,fastllm::prefix_v4::RecordHeader&,std::string&,std::string&);
static Json One(ReadFn read,const Json&c,size_t cap,int error,bool direct=false){
 std::string out="unchanged-sentinel",why;fastllm::prefix_v4::RecordHeader h;fault=Fault{cap,error,false};trace=IoTrace{};bool ok=false;
 try{ok=read(c["path"].string_value(),uint64_t(c["offset"].number_value()),h,out,why);}catch(const std::exception&e){why=e.what();}
 Require(trace.alignmentErrors==0,"direct_alignment");
 if(direct)Require(trace.bufferedOpens==0&&trace.bufferedReads==0&&trace.directOpens>0,"direct_fd_no_fallback");
 else Require(trace.directOpens==0&&trace.directReads==0,"buffered_default_unchanged");
 bool injected=fault.fired;fault=Fault{};auto expected=c["error"].string_value();if((error==EIO||error==EINVAL)&&injected)expected="v4_log_read";
 Require(ok==expected.empty(),"status:"+c["name"].string_value()+":"+why);Require(why==expected,"error:"+c["name"].string_value()+":"+why+":"+expected);
 if(expected.empty()||expected=="v4_control_record_digest"){auto bytes=Bytes(size_t(c["payload_bytes"].number_value()));if(c["name"].string_value()=="bad-payload")bytes[7]^=1;Require(out==bytes,"independent_byte_oracle");}
 else Require(out=="unchanged-sentinel","failed_read_modified_output");
 return Json::object{{"ok",ok},{"error",why},{"out",double(out.size())},{"payload",double(h.payloadBytes)},{"physical",double(h.physicalBytes)},{"injected",injected}};
}
static uint64_t PhysicalReadBytes(){std::ifstream f("/proc/self/io");std::string k;uint64_t v;while(f>>k>>v)if(k=="read_bytes:")return v;throw std::runtime_error("proc_io");}
int main(int argc,char**argv){
 try{
  Require(argc==2,"usage: record_direct_t0 recorded-inputs.json");auto cfg=Config(argv[1]);size_t before=Fds(),controls=0,ranges=0;
  for(const auto&c:cfg["records"].array_items()){
   std::vector<std::pair<size_t,int>> modes={{0,0},{0,EINTR},{0,EIO},{0,EINVAL}};
   if(c["payload_bytes"].number_value()<16384){modes.push_back({37,0});modes.push_back({127,EINTR});modes.push_back({4097,0});}else modes.push_back({8192,EINTR});
   for(auto mode:modes){
    setenv("FASTLLM_PREFIX_CACHE_DIRECT_IO","0",1);auto a=One(Baseline::V4ReadRecord,c,mode.first,mode.second);auto b=One(Buffered::V4ReadRecord,c,mode.first,mode.second);
    setenv("FASTLLM_PREFIX_CACHE_DIRECT_IO","1",1);auto d=One(Direct::V4ReadRecord,c,mode.first,mode.second,true);
    Require(a==b&&a==d,"buffered_direct_record_pair:"+c["name"].string_value());++controls;
   }
  }
  auto large=*std::find_if(cfg["records"].array_items().begin(),cfg["records"].array_items().end(),[](const Json&c){return c["name"].string_value()=="valid-1048576";});
  fs::path path=large["path"].string_value();
  for(uint64_t at:{0ull,1ull,127ull,4095ull,4096ull,8193ull})for(size_t n:{size_t(0),size_t(1),size_t(128),size_t(4095),size_t(4096),size_t(8193)}){
   fault=Fault{};auto a=Baseline::V4ReadRange(path,at,n);trace=IoTrace{};auto d=Direct::V4ReadRange(path,at,n);
   Require(a==d,"unaligned_logical_range");Require(trace.alignmentErrors==0&&trace.bufferedOpens==0&&trace.bufferedReads==0,"range_direct_only");++ranges;
  }
  for(int error:{EINVAL,EOPNOTSUPP}){
   fault=Fault{};fault.openError=error;trace=IoTrace{};std::string out="unchanged",why;fastllm::prefix_v4::RecordHeader h;bool ok=false;
   try{ok=Direct::V4ReadRecord(path,4096,h,out,why);}catch(const std::exception&e){why=e.what();}
   Require(!ok&&why=="v4_log_open"&&out=="unchanged"&&fault.fired&&trace.bufferedOpens==0&&trace.bufferedReads==0,"unsupported_must_not_fallback");fault=Fault{};
  }
  bool bounds=false;try{Direct::V4ReadRange(path,UINT64_MAX,4096);}catch(const std::exception&e){bounds=std::string(e.what())=="v4_direct_io_bounds";}Require(bounds,"offset_overflow");
  One(Baseline::V4ReadRecord,large,0,0);uint64_t ioBefore=PhysicalReadBytes();One(Direct::V4ReadRecord,large,0,0,true);uint64_t physical=PhysicalReadBytes()-ioBefore;
  uint64_t recordPhysical=(128+uint64_t(large["payload_bytes"].number_value())+4095)/4096*4096;
  Require(physical>=recordPhysical,"warm_file_direct_physical_io");Require(Fds()==before,"fd_leak");
  std::printf("PASS record_direct controls=%zu ranges=%zu; independent bytes/errors/short-header/payload/padding/EINTR/partial/EIO/EINVAL; fd O_DIRECT/alignment/no fallback; warm_file_physical_read_bytes=%llu record_physical_bytes=%llu direct_read_calls=%llu direct_read_bytes=%llu\n",controls,ranges,(unsigned long long)physical,(unsigned long long)recordPhysical,(unsigned long long)Direct::V4Split().directReadCalls.load(),(unsigned long long)Direct::V4Split().directReadBytes.load());return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL record_direct_t0: %s\n",e.what());return 1;}
}
