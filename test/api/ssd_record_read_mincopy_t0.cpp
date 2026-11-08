#include "utils/disk_prefix_format_v4.h"
#include "json11.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
using Json=json11::Json;
namespace fs=std::filesystem;
static void Require(bool ok,const std::string&s){if(!ok)throw std::runtime_error(s);}
struct Fault{size_t cap=0;int error=0;bool fired=false;};static Fault fault;
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __real_preadv(int,const iovec*,int,off_t);
static bool Inject(size_t n){if(fault.error&&!fault.fired&&n>128){fault.fired=true;errno=fault.error;return true;}return false;}
extern "C" ssize_t __wrap_pread(int fd,void*out,size_t n,off_t at){if(Inject(n))return -1;return __real_pread(fd,out,fault.cap?std::min(n,fault.cap):n,at);}
extern "C" ssize_t __wrap_preadv(int fd,const iovec*v,int count,off_t at){
 size_t total=0;for(int i=0;i<count;++i)total+=v[i].iov_len;if(Inject(total))return -1;
 if(!fault.cap)return __real_preadv(fd,v,count,at);
 iovec limited[3];int n=0;size_t left=fault.cap;
 for(int i=0;i<count&&left;++i){limited[n]=v[i];limited[n].iov_len=std::min(left,v[i].iov_len);left-=limited[n].iov_len;++n;}
 return __real_preadv(fd,limited,n,at);
}
namespace Baseline{namespace prefix_v4=fastllm::prefix_v4;using namespace prefix_v4;
static void Check(bool ok,const std::string&s){Require(ok,s);}
#include "record_base.inc"
}
namespace Candidate{namespace prefix_v4=fastllm::prefix_v4;using namespace prefix_v4;
static void Check(bool ok,const std::string&s){Require(ok,s);}
#include "record_candidate.inc"
}
static std::string Bytes(size_t n){std::string s(n,'\0');for(size_t i=0;i<n;++i)s[i]=char((i*31+7)%251);return s;}
static Json Config(const char*p){std::ifstream f(p);Require(bool(f),"fixture_open");std::string text((std::istreambuf_iterator<char>(f)),{}),error;auto j=Json::parse(text,error);Require(error.empty(),"fixture_json");return j;}
static size_t Fds(){size_t n=0;for(const auto&p:fs::directory_iterator("/proc/self/fd")){(void)p;++n;}return n;}
static void Prepare(const Json&cfg){
 using namespace fastllm::prefix_v4;const fs::path dir=cfg["record_dir"].string_value();Json::array files;
 auto save=[&](const std::string&name,const std::vector<char>&raw,size_t n,const std::string&error){
  auto p=dir/(name+".bin");std::ofstream f(p,std::ios::binary);Require(bool(f),"fixture_write");f.write(raw.data(),raw.size());f.close();
  files.push_back(Json::object{{"name",name},{"path",p.string()},{"payload_bytes",double(n)},{"error",error},{"offset",4096}});
 };
 for(const auto&value:cfg["sizes"].array_items()){
  size_t n=value.int_value();auto body=Bytes(n);RecordHeader h;h.type=kRecordMetadata;h.coverageMode=kCoverageFullRecord;h.payloadBytes=n;h.physicalBytes=(128+n+4095)/4096*4096;h.sequence=17;h.digest=RecordDigest(h.type,body);
  auto header=EncodeRecordHeader(h);std::vector<char> raw(4096,'P');raw.insert(raw.end(),header.begin(),header.end());raw.insert(raw.end(),body.begin(),body.end());raw.resize(4096+h.physicalBytes,'Z');save("valid-"+std::to_string(n),raw,n,"");
  if(n!=8193)continue;
  auto x=raw;x.resize(4096+127);save("short-header",x,n,"v4_log_short_read");
  x=raw;x.resize(4096+128+n-1);save("short-payload",x,n,"v4_log_short_read");
  x=raw;x.resize(x.size()-1);save("short-padding",x,n,"v4_log_short_read");
  x=raw;x[4096]^=1;save("bad-magic",x,n,"v4_record_magic");
  x=raw;x[4116]^=1;save("bad-header-crc",x,n,"v4_record_header_crc");
  x=raw;x[4096+128+7]^=1;save("bad-payload",x,n,"v4_control_record_digest");
  auto bad=h;bad.digest=std::string(64,'0');header=EncodeRecordHeader(bad);x=raw;std::copy(header.begin(),header.end(),x.begin()+4096);save("bad-digest",x,n,"v4_control_record_digest");
  bad=h;bad.physicalBytes+=4096;header=EncodeRecordHeader(bad);x=raw;std::copy(header.begin(),header.end(),x.begin()+4096);save("bad-physical",x,n,"v4_record_physical");
  bad=h;bad.payloadBytes=kMaxPackedPayloadBytes+1;header=EncodeRecordHeader(bad);x=raw;std::copy(header.begin(),header.end(),x.begin()+4096);save("bad-bounds",x,n,"v4_record_payload_bounds");
  bad=h;bad.flags=0x80000000u;header=EncodeRecordHeader(bad);x=raw;std::copy(header.begin(),header.end(),x.begin()+4096);save("bad-flags",x,n,"v4_record_unknown_flag");
 }
 std::ofstream f(cfg["record_manifest"].string_value());f<<Json(Json::object{{"records",files}}).dump_compact()<<"\n";std::puts("Prepared record fixtures; no tests executed");
}
using ReadFn=bool(*)(const fs::path&,uint64_t,fastllm::prefix_v4::RecordHeader&,std::string&,std::string&);
static Json One(ReadFn read,const Json&c,size_t cap,int error){
 std::string out="unchanged-sentinel",why;fastllm::prefix_v4::RecordHeader h;fault=Fault{cap,error,false};bool ok=false;
 try{ok=read(c["path"].string_value(),uint64_t(c["offset"].number_value()),h,out,why);}catch(const std::exception&e){why=e.what();}
 bool injected=fault.fired;fault=Fault{};auto expected=c["error"].string_value();if(error==EIO&&injected)expected="v4_log_read";
 Require(ok==expected.empty(),"status:"+c["name"].string_value()+":"+why);Require(why==expected,"error:"+c["name"].string_value()+":"+why+":"+expected);
 if(expected.empty()||expected=="v4_control_record_digest"){auto bytes=Bytes(size_t(c["payload_bytes"].number_value()));if(c["name"].string_value()=="bad-payload")bytes[7]^=1;Require(out==bytes,"independent_byte_oracle");}
 else Require(out=="unchanged-sentinel","failed_read_modified_output");
 return Json::object{{"ok",ok},{"error",why},{"out",double(out.size())},{"payload",double(h.payloadBytes)},{"physical",double(h.physicalBytes)},{"injected",injected}};
}
int main(int argc,char**argv){
 try{Require(argc==3,"usage: record_t0 input prepare|controls");auto cfg=Config(argv[1]);if(std::string(argv[2])=="prepare"){Prepare(cfg);return 0;}
  size_t before=Fds(),count=0;
  for(const auto&c:cfg["records"].array_items()){std::vector<std::pair<size_t,int>> modes={{0,0},{0,EINTR},{0,EIO}};
   if(c["payload_bytes"].number_value()<16384){modes.push_back({37,0});modes.push_back({127,EINTR});modes.push_back({4097,0});}else modes.push_back({8192,EINTR});
   for(auto mode:modes){auto a=One(Baseline::V4ReadRecord,c,mode.first,mode.second),b=One(Candidate::V4ReadRecord,c,mode.first,mode.second);Require(a==b,"record_source_pair");++count;}}
  Require(Fds()==before,"fd_leak");std::printf("PASS record controls=%zu, independent bytes, short/padding/EINTR/partial/EIO, fd cleanup\n",count);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL record_t0: %s\n",e.what());return 1;}
}
