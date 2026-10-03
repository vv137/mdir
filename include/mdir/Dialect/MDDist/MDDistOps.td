// Restricted fixed-layout host execution plans.
#ifndef MDIR_MDDIST_OPS_TD
#define MDIR_MDDIST_OPS_TD
include "mdir/Dialect/MDRT/MDRTTypes.td"
include "mlir/IR/SymbolInterfaces.td"
def MDDist_Dialect : Dialect {
  let name = "md_dist";
  let cppNamespace = "::mdir::md_dist";
  let summary = "Distribution execution contracts";
  let dependentDialects = ["::mdir::mdrt::MDRTDialect"];
}
class MDDist_Op<string name, list<Trait> traits = []> : Op<MDDist_Dialect, name, traits>;
def MDDist_PlanOp : MDDist_Op<"reference_plan", [SingleBlock, NoTerminator]> {
  let summary = "An evaluation-local LJ and optional bonded reference plan";
  let description = [{
    The host binds an immutable checked owner/replica map and disjoint center
    subsets for one evaluation. The body schedules a halo and two dispatches
    to the same existing md_exec kernel. An optional third block argument
    binds a topology map, with one bond dispatch and reverse completion.
    This restricted reference bridge is not a general distributed field IR.
    Unknown operations are rejected.
  }];
  let arguments = (ins DenseI64ArrayAttr:$grid);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$grid attr-dict-with-keyword $body";
  let hasRegionVerifier = 1;
}
def MDDist_HaloStartOp : MDDist_Op<"halo_start", [HasParent<"PlanOp">]> {
  let summary = "Start the bound map's coordinate transfer";
  let arguments = (ins MDRT_LayoutType:$layout, MDRT_TransferMapType:$map);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = "$layout `via` $map attr-dict `:` qualified(type($layout)) `,` qualified(type($map)) `->` type($event)";
}
def MDDist_HaloWaitOp : MDDist_Op<"halo_wait", [HasParent<"PlanOp">]> {
  let summary = "Complete transfer, unpack, and release send storage";
  let arguments = (ins MDRT_EventType:$event);
  let assemblyFormat = "$event attr-dict `:` type($event)";
}
def MDDist_DispatchOp : MDDist_Op<"dispatch", [HasParent<"PlanOp">]> {
  let summary = "Invoke the compiled pair kernel on a checked center subset";
  let arguments = (ins FlatSymbolRefAttr:$callee, StrAttr:$subset);
  let assemblyFormat = "$callee $subset attr-dict";
}
// Optional topology extension of the same restricted reference plan.
def MDDist_TopologyStartOp : MDDist_Op<"topology_start", [HasParent<"PlanOp">]> {
  let summary = "Start coordinate transfer on the topology route";
  let arguments = (ins MDRT_LayoutType:$layout, MDRT_TransferMapType:$map);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = "$layout `via` $map attr-dict `:` qualified(type($layout)) `,` qualified(type($map)) `->` type($event)";
}
def MDDist_TopologyWaitOp : MDDist_Op<"topology_wait", [HasParent<"PlanOp">]> {
  let summary = "Make topology coordinates readable and release sends";
  let arguments = (ins MDRT_EventType:$event);
  let assemblyFormat = "$event attr-dict `:` type($event)";
}
def MDDist_BondDispatchOp : MDDist_Op<"bond_dispatch", [HasParent<"PlanOp">]> {
  let summary = "Evaluate unique bonds into separate contribution rows";
  let arguments = (ins FlatSymbolRefAttr:$callee);
  let assemblyFormat = "$callee attr-dict";
}
def MDDist_ReverseStartOp : MDDist_Op<"reverse_start", [HasParent<"PlanOp">]> {
  let summary = "Start return of bonded contributions on the topology route";
  let arguments = (ins MDRT_LayoutType:$layout, MDRT_TransferMapType:$map);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = "$layout `via` $map attr-dict `:` qualified(type($layout)) `,` qualified(type($map)) `->` type($event)";
}
def MDDist_ReverseWaitOp : MDDist_Op<"reverse_wait", [HasParent<"PlanOp">]> {
  let summary = "Complete and apply bonded contributions to their owners";
  let arguments = (ins MDRT_EventType:$event);
  let assemblyFormat = "$event attr-dict `:` type($event)";
}
#endif
