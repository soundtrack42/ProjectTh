"""LossScore weight sensitivity (Appendix app:weights). Set C to the folder holding run{1,2,3}_{syn,real} benchmark outputs."""
import csv, collections, numpy as np, itertools
C="/tmp/claude-1000/-home-giorge-Desktop-ProjectTh/da189864-ada5-4cdd-ba02-15aaaeb1a83b/scratchpad/cnt5"
W0=np.array([1.0,15.0,2.0,0.025]); K=["abs_sched_err_s","abs_stop_err_m","step_smooth_cost","spec_energy_wh_tkm"]
suites={"syn":["run{r}_syn/synthetic_baseline_metrics.csv","run{r}_syn/synthetic_extreme_metrics.csv"],
        "real_meas":["run{r}_real/real_world_api_baseline_metrics.csv"],"real_ext":["run{r}_real/real_world_api_extreme_metrics.csv"]}
rng=np.random.default_rng(0)
for s,files in suites.items():
    T=collections.defaultdict(list)
    for r in (1,2,3):
        for f in files:
            for row in csv.DictReader(open(f"{C}/{f.format(r=r)}")):
                T[row["method"]].append([float(row[k]) for k in K])
    names=sorted(T,key=lambda n:int(n.split(".")[0])); X=np.array([np.mean(T[n],0) for n in names])
    ours=np.array([int(n.split(".")[0]) in range(2,8) for n in names])
    sc=X@W0; order=np.argsort(sc)
    print(f"\n== {s}: n journeys/run={len(T[names[0]])//3}")
    print(" nominal top5:",[names[i].split('.')[0] for i in order[:5]], np.round(sc[order[:5]],3))
    contrib=X*W0; spread=contrib.max(0)-contrib.min(0)
    print(" term contribution mean:",np.round(contrib.mean(0),3)," range across methods:",np.round(spread,3))
    top=order[:8]; sp2=contrib[top].max(0)-contrib[top].min(0); print(" range among 8 best:",np.round(sp2,3))
    def stats(Ws):
        win_ours=0; beat=0; taus=[]
        for w in Ws:
            v=X@w; o=np.argsort(v)
            win_ours+=ours[o[0]]; beat+= v[ours].min()<v[~ours].min()
            # spearman
            rk=np.empty(len(v)); rk[o]=np.arange(len(v)); rk0=np.empty(len(v)); rk0[order]=np.arange(len(v))
            taus.append(np.corrcoef(rk,rk0)[0,1])
        return 100*win_ours/len(Ws), np.mean(taus), np.min(taus)
    for i,k in enumerate(K):
        for f in (0,0.25,0.5,2,4):
            w=W0.copy(); w[i]*=f; v=X@w; o=np.argsort(v)
            print(f"  {k[:10]:10s} x{f:<4}: best={names[o[0]].split('.')[0]:>2} best_ours={min(v[ours]):.3f} best_base={min(v[~ours]):.3f} ({names[np.where(~ours)[0][np.argmin(v[~ours])]].split('.')[0]})")
    Ws=W0*np.exp(rng.uniform(np.log(1/4),np.log(4),size=(20000,4)))
    print(" random x[1/4,4] each: ours-first %.1f%%, spearman mean %.3f min %.3f"%stats(Ws))
    Ws=W0*np.exp(rng.uniform(np.log(1/10),np.log(10),size=(20000,4)))
    print(" random x[1/10,10] each: ours-first %.1f%%, spearman mean %.3f min %.3f"%stats(Ws))
    # dominance: any baseline better-or-equal on all 4 than best ours?
    for j in np.where(ours)[0]:
        dom=[names[i].split('.')[0] for i in np.where(~ours)[0] if np.all(X[i]<=X[j])]
        if dom: print("  method",names[j].split('.')[0],"dominated by",dom)
    print("  means:"); [print("   ",n[:28].ljust(28),np.round(X[i],4),round(sc[i],3)) for i,n in enumerate(names)]
