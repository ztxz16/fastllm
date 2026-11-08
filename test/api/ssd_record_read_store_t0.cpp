#define main OriginalFullStoreTests
#include "../basic/test_disk_prefix_store_v4.cpp"
#undef main
int main(){
 const auto tmp=TempRoot();
 try{
  writeReadRecover(tmp/"basic");sqliteCatalogLossRebuildsFromLog(tmp/"index-loss");
  unfinishedTailPublishesNothing(tmp/"tail");corruptLogFailsClosed(tmp/"corrupt");
  quotaEvictionAndCompaction(tmp/"quota");v4CacheInterface(tmp/"interface");
  strictVerifyOffStillSafe(tmp/"strict");rebuildRejectsCorruptPayload(tmp/"rebuild-corrupt");
  writeRestartReadBackExact(tmp/"exact");oddSizedObjectsStayReadable(tmp/"odd");
  routeBMultiBlockChainRestoresByteExact(tmp/"routeb");
  routeBRebuildIndexRestoresBlocksProjection(tmp/"routeb-index");
  routeBCompactKeepsReachableBlocksReclaimsUnreferenced(tmp/"routeb-compact");
 }catch(const std::exception&e){std::cerr<<"FAIL store-read controls: "<<e.what()<<"\n";return 1;}
 if(failures)return 1;
 std::puts("PASS 13 relevant real store read/write/rebuild/compact controls");return 0;
}
