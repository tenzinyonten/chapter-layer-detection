"""
Leakage check for the Chapter split: verbatim shingle matching. Read-only.

Usage:
    python src/check_chapter_leakage.py
"""
import csv,re
from pathlib import Path
from collections import defaultdict,Counter
R=Path(__file__).resolve().parents[1]
aud={r['pecha_id']:r for r in csv.DictReader(open(R/'data/tsawa_audit.csv',encoding='utf-8'))}
split={r['pecha_id']:r['split'] for r in csv.DictReader(l for l in open(R/'data/split_frozen.csv',encoding='utf-8') if not l.startswith('#'))}
spans=defaultdict(list)
for r in csv.DictReader(open(R/'data/chapter_spans_clean.csv',encoding='utf-8')):
    if r['dropped']!='True' and r['pecha_id'] in split: spans[r['pecha_id']].append((int(r['start']),int(r['end'])))
books=sorted(split)
WS=re.compile(r'\s+');SYL=re.compile(r'[་-༔\s]+')
syls=lambda s:[x for x in SYL.split(s) if x]
K=6;MIN=40
tx={p:Path(aud[p]['base_path']).read_text(encoding='utf-8') for p in books}
st={p:WS.sub('',tx[p]) for p in books}
needles={p:[] for p in books}; allsp={p:[] for p in books}
for p in books:
    for s,e in spans[p]:
        nd=WS.sub('',tx[p][s:e]); allsp[p].append(nd)
        if e-s>=MIN and len(syls(nd))>=K+2: needles[p].append(nd)
idx=defaultdict(list)
for p,ns in needles.items():
    for i,nd in enumerate(ns): idx[hash(tuple(syls(nd)[1:1+K]))].append((p,i))
found=defaultdict(set)
for p2 in books:
    sy=syls(st[p2]);c=set()
    for j in range(len(sy)-K+1):
        h=hash(tuple(sy[j:j+K]))
        if h in idx:c.update(idx[h])
    for p,i in c:
        if p!=p2 and needles[p][i] in st[p2]: found[(p,i)].add(p2)
train={p for p in books if split[p]=='train'}
print("== long spans (>=40 chars) found verbatim in another book ==")
for sp in ('train','val','test'):
    n=h_tr=h_any=0
    for p in books:
        if split[p]!=sp:continue
        for i in range(len(needles[p])):
            n+=1;o=found.get((p,i),set())
            h_any+=bool(o);h_tr+=bool(o&train) if sp!='train' else bool(o-{p})
    lab='in another train book' if sp=='train' else 'in a train book'
    print(f"{sp:5} long={n:4}  in any other book={h_any:3} ({100*h_any/max(n,1):.1f}%)  {lab}={h_tr:3} ({100*h_tr/max(n,1):.1f}%)")
# per-book contribution in test/val
for sp in ('val','test'):
    c=Counter()
    for p in books:
        if split[p]!=sp:continue
        for i in range(len(needles[p])):
            if found.get((p,i),set())&train:c[p]+=1
    print(sp,'top books by leaked long spans:',c.most_common(5))
print("\n== exact text of ANY span (incl. short) also a Chapter span in a train book ==")
trs=Counter(nd for p in train for nd in allsp[p])
for sp in ('val','test'):
    tot=hit=0;ex=Counter()
    for p in books:
        if split[p]!=sp:continue
        for nd in allsp[p]:
            tot+=1
            if nd in trs:hit+=1;ex[nd]+=1
    print(f"{sp:5} spans={tot} exact-title match in train={hit} ({100*hit/tot:.1f}%)  top: {[(k[:20],v) for k,v in ex.most_common(4)]}")
links=set()
for (p,i),o in found.items():
    for q in o: links.add(tuple(sorted((p,q))))
print("\nbook pairs sharing >=1 long span:",len(links),"; straddling splits:",sum(split[a]!=split[b] for a,b in links))
