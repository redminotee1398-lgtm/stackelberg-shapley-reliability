import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

class Subsystem:
    def __init__(self,name,alpha,beta,pi,Rmin=0.90,Rmax=0.9995,role='follower'):
        self.name=name; self.alpha=alpha; self.beta=beta; self.pi=pi
        self.Rmin=Rmin; self.Rmax=Rmax; self.role=role
    def cost(self,R):
        R=np.clip(R,self.Rmin,min(self.Rmax,0.999999))
        return self.alpha*np.exp(self.beta*R/(1-R))
    def dcost(self,R):
        R=np.clip(R,self.Rmin,min(self.Rmax,0.999999))
        return self.cost(R)*self.beta/(1-R)**2

class SeriesSystem:
    def __init__(self,subs,L=0):
        self.subs=subs; self.n=len(subs); self.L=L; self.F=[i for i in range(self.n) if i!=L]
    def R_sys(self,R): return float(np.prod(R))
    def welfare(self,R): return sum(s.pi for s in self.subs)*self.R_sys(R)-sum(s.cost(R[i]) for i,s in enumerate(self.subs))
    def utility(self,i,R): return self.subs[i].pi*self.R_sys(R)-self.subs[i].cost(R[i])
    def _best(self,i,R,stake):
        s=self.subs[i]; target=stake*np.prod([R[j] for j in range(self.n) if j!=i]); lo=s.Rmin; hi=min(s.Rmax,0.999999)
        if s.dcost(lo)>=target: return lo
        if s.dcost(hi)<=target: return hi
        return brentq(lambda x:s.dcost(x)-target,lo,hi,xtol=1e-12)
    def _nash_eff(self,stakes,fixed=None):
        R=np.array([s.Rmin for s in self.subs],float)
        if fixed:
            for k,v in fixed.items(): R[k]=v
        players=[i for i in range(self.n) if not fixed or i not in fixed]
        for _ in range(1000):
            old=R.copy()
            for i in players: R[i]=self._best(i,R,stakes[i])
            if np.max(np.abs(R-old))<1e-11: break
        return R
    def solve_nash(self): return self._nash_eff([s.pi for s in self.subs])
    def solve_social(self):
        P=sum(s.pi for s in self.subs); return self._nash_eff([P]*self.n)
    def followers_nash(self,RL): return self._nash_eff([s.pi for s in self.subs],{self.L:RL})
    def solve_stackelberg(self,theta=0.70):
        lo=self.subs[self.L].Rmin; hi=min(self.subs[self.L].Rmax,0.999999)
        def obj(RL):
            R=self.followers_nash(RL)
            return theta*self.welfare(R)+(1-theta)*self.utility(self.L,R)
        res=minimize_scalar(lambda x:-obj(x),bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
        return self.followers_nash(res.x)
    def followers_subsidized(self,RL,s_sub):
        P=sum(s.pi for s in self.subs)
        eff=[s.pi for s in self.subs]
        for j in self.F:
            eff[j]=self.subs[j].pi+s_sub*(P-self.subs[j].pi)
        return self._nash_eff(eff,{self.L:RL})
    def solve_stackelberg_follower_subsidy(self,s_sub,theta=0.70):
        lo=self.subs[self.L].Rmin; hi=min(self.subs[self.L].Rmax,0.999999)
        def obj(RL):
            R=self.followers_subsidized(RL,s_sub)
            return theta*self.welfare(R)+(1-theta)*self.utility(self.L,R)
        res=minimize_scalar(lambda x:-obj(x),bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
        return self.followers_subsidized(res.x,s_sub)

def eps():
    return SeriesSystem([
        Subsystem('Solar-Array Operating (L)',0.024,0.088,3.2,role='leader'),
        Subsystem('Battery / Cell',0.032,0.078,1.9),
        Subsystem('Electrical Distribution',0.027,0.071,1.5),
        Subsystem('Solar-Array Deployment',0.020,0.060,1.1),
    ])

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif',
    'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'axes.grid':True,
    'grid.color':'#DDDDDD','grid.linewidth':0.6,'legend.frameon':True,'legend.framealpha':0.92,
})
C={'leader':'#C0392B','stk':'#2471A3','soc':'#1E8449','nash':'#7F8C8D'}

# Figure 11: corrected dynamic residual-gap percentage
g=eps(); thetas=np.linspace(0,1,21)
Rsys=[]; RL=[]; RF=[]
for th in thetas:
    R=g.solve_stackelberg(theta=float(th)); Rsys.append(g.R_sys(R)); RL.append(R[g.L]); RF.append(np.mean(R[1:]))
Rn=g.R_sys(g.solve_nash()); Ro=g.R_sys(g.solve_social())
resid=100*(Ro-Rsys[-1])/(Ro-Rn)
fig,ax=plt.subplots(figsize=(6.4,4.2666667),dpi=240)
ax.plot(thetas,Rsys,color=C['leader'],lw=2.6,marker='o',ms=4,label=r'system reliability $R_{\mathrm{sys}}^*$')
ax.plot(thetas,RL,color=C['stk'],lw=2.0,marker='s',ms=3.5,label=r'leader reliability $R_L^*$')
ax.plot(thetas,RF,color=C['soc'],lw=2.0,ls='--',marker='^',ms=3.5,label=r'mean follower reliability $\bar R_F^*$')
ax.axhline(Ro,color='#444',ls=':',lw=1.4); ax.text(0.02,Ro+0.004,f'social-optimum $R_{{\\mathrm{{sys}}}}={Ro:.3f}$',fontsize=8.6,color='#444')
ax.axhline(Rn,color=C['nash'],ls=':',lw=1.2); ax.text(0.02,Rn-0.009,f'Nash $R_{{\\mathrm{{sys}}}}={Rn:.3f}$',fontsize=8.6,color=C['nash'])
ax.annotate('',xy=(0.965,Ro),xytext=(0.965,Rsys[-1]),arrowprops=dict(arrowstyle='<->',color='#444',lw=1.1))
ax.text(0.95,(Ro+Rsys[-1])/2,f'residual\nreliability gap ({resid:.1f}%)',fontsize=8.0,color='#444',ha='right',va='center')
ax.set_xlabel(r'leader system-accountability weight $\theta$'); ax.set_ylabel('reliability at equilibrium')
ax.set_title('Leader accountability helps, but cannot close the gap alone'); ax.legend(fontsize=8.2,loc='center',bbox_to_anchor=(0.43,0.38))
ax.set_xlim(0,1); ax.set_ylim(0.76,0.97); fig.tight_layout(); fig.savefig('/mnt/data/aej_sync/figure11_corrected.png',dpi=240,facecolor='white'); plt.close(fig)

# Figure 15: Stackelberg + follower-only subsidy, synthetic chains
rng=np.random.default_rng(7); grid=np.linspace(0,1,11); curves={}
for n in (4,8,12):
    ss=[Subsystem('L',0.024,0.088,3.2,role='leader')]
    for k in range(n-1): ss.append(Subsystem(f'F{k}',0.028,0.072,float(1.0+0.6*rng.random())))
    gg=SeriesSystem(ss)
    Wn=gg.welfare(gg.solve_nash()); Wo=gg.welfare(gg.solve_social())
    vals=[]
    for s in grid:
        R=gg.solve_stackelberg_follower_subsidy(float(s),theta=0.70)
        vals.append(100*(gg.welfare(R)-Wn)/(Wo-Wn))
    curves[n]=np.array(vals)
fig,ax=plt.subplots(figsize=(6.4,4.2666667),dpi=240)
cols={4:'#C0392B',8:'#2471A3',12:'#1E8449'}
for n in (4,8,12):
    ax.plot(grid,curves[n],color=cols[n],lw=2.4,marker='o',ms=3.5,label=fr'leader + follower subsidy, $n={n}$')
ax.set_xlabel(r'follower subsidy intensity $s$ (0 = leader only; 1 = full follower subsidy)', fontsize=9.5)
ax.set_ylabel(r'Nash$\rightarrow$social welfare gap closed (%)')
ax.set_title('Follower-side subsidies complement the accountable leader', fontsize=11.5)
ax.axhline(100,color='#999',ls=':',lw=1.0); ax.legend(fontsize=8.1,loc='lower right'); ax.set_ylim(0,108); ax.set_xlim(-0.02,1.02)
fig.tight_layout(); fig.savefig('/mnt/data/aej_sync/figure15_corrected.png',dpi=240,facecolor='white'); plt.close(fig)

print('Figure 11 residual gap:',resid)
for n,v in curves.items(): print('n',n, np.round(v,2).tolist())
