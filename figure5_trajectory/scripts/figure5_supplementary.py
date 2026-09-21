"""# Figure 5 supplementary figures

This notebook-style analysis validates the authoritative Figure 5 caches, optionally
refreshes the five-tool trajectory benchmark, and draws Supplementary Figures 12-14.
The default cached mode never opens the atlas or ATAC fragments and never invokes R.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zlib
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, mannwhitneyu, spearmanr
from supplementary_style import (
    PAGE_WIDTH_IN, PAGE_HEIGHT_IN, EXPORT_DPI, standardize_figure,
)


# ## 1. Paths and the one execution switch
# Cached mode is the everyday figure-export path. Refresh mode is reserved for
# rebuilding the multi-environment benchmark from the immutable RNA atlas.

REFRESH_EXPENSIVE_RESULTS = False

FIGURE5_DIRECTORY = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = FIGURE5_DIRECTORY / "input"
RESULT_DIRECTORY = FIGURE5_DIRECTORY / "result"
CACHE_DIRECTORY = RESULT_DIRECTORY / "cache"
SUPPLEMENTARY_CACHE_DIRECTORY = CACHE_DIRECTORY / "supplementary"
PDF_DIRECTORY = RESULT_DIRECTORY / "pdf"
QC_DIRECTORY = RESULT_DIRECTORY / "qc"

ATLAS_PATH = INPUT_DIRECTORY / "HumanFetalRetina.h5ad"
HARMONIZATION_PATH = INPUT_DIRECTORY / "hector_label_harmonization.csv"
MAIN_CELL_CACHE_PATH = CACHE_DIRECTORY / "hector_cells.parquet"
MAIN_CACHE_MANIFEST_PATH = CACHE_DIRECTORY / "cache_manifest.json"
RNA_CURVE_PATH = CACHE_DIRECTORY / "rna_curves.csv"
SIGNAC_PROFILE_PATH = CACHE_DIRECTORY / "signac_profiles.csv.gz"
SIGNAC_EFFECT_PATH = CACHE_DIRECTORY / "signac_effects.csv"
TABIX_EFFECT_PATH = CACHE_DIRECTORY / "tabix_effects.csv"

BENCHMARK_CELL_PATH = SUPPLEMENTARY_CACHE_DIRECTORY / "benchmark_cells.parquet"
BENCHMARK_ORDERING_PATH = SUPPLEMENTARY_CACHE_DIRECTORY / "benchmark_orderings.parquet"
LINEAGE_AGREEMENT_PATH = SUPPLEMENTARY_CACHE_DIRECTORY / "lineage_agreement.csv"
PRECURSOR_ORDER_PATH = SUPPLEMENTARY_CACHE_DIRECTORY / "precursor_order.csv"
SUPPLEMENTARY_MANIFEST_PATH = SUPPLEMENTARY_CACHE_DIRECTORY / "manifest.json"
SUPPLEMENTARY_REFRESH_LOG_PATH = QC_DIRECTORY / "supplementary_refresh.log"
SUPPLEMENTARY_PARITY_PATH = QC_DIRECTORY / "supplementary_parity_report.json"

S12_PDF_PATH = PDF_DIRECTORY / "supplementary_figure_12_retina_validation.pdf"
S12_PNG_PATH = PDF_DIRECTORY / "supplementary_figure_12_retina_validation.png"
S13_PDF_PATH = PDF_DIRECTORY / "supplementary_figure_13_retina_subtypes.pdf"
S13_PNG_PATH = PDF_DIRECTORY / "supplementary_figure_13_retina_subtypes.png"
S14_PDF_PATH = PDF_DIRECTORY / "supplementary_figure_14_rna_chromatin.pdf"
S14_PNG_PATH = PDF_DIRECTORY / "supplementary_figure_14_rna_chromatin.png"

# The refresh runs its five trajectory tools outside the HECTOR environment,
# because CytoTRACE 2 pins NumPy below 2 and Monocle 3 brings its own R, either
# of which would break HECTOR's own stack. One conda environment holds all five:
# build it from environments/traj.yml, which carries DPT (through scanpy),
# Palantir and CytoTRACE 2 on Python 3.11, and Monocle 3 and Slingshot on R
# 4.4.3. Its Python and its Rscript are found from the conda installation that is
# running this file, and either can be pointed somewhere else by exporting
# FIGURE5_TRAJPY_PYTHON or FIGURE5_TRAJ_RSCRIPT. The two variables stay separate
# so that a split installation still works.

TRAJECTORY_ENVIRONMENT_NAME = "traj"


def _conda_environment_root() -> Path:
    """Directory that holds the sibling conda environments of this interpreter."""
    told = os.environ.get("CONDA_ENVS_PATH") or os.environ.get("CONDA_ENVS_DIRS")
    if told:
        return Path(told.split(os.pathsep)[0])
    return Path(sys.executable).resolve().parent.parent.parent


TRAJPY_PYTHON = Path(
    os.environ.get("FIGURE5_TRAJPY_PYTHON")
    or _conda_environment_root() / TRAJECTORY_ENVIRONMENT_NAME / "bin" / "python"
)
TRAJ_RSCRIPT = Path(
    os.environ.get("FIGURE5_TRAJ_RSCRIPT")
    or _conda_environment_root() / TRAJECTORY_ENVIRONMENT_NAME / "bin" / "Rscript"
)

SUPPLEMENTARY_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
QC_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ## 2. Frozen biological and benchmark settings
# These values reproduce the validated comparison. Each method follows its own
# documented preprocessing, and donor is deliberately not regressed out.

BENCHMARK_SETTINGS = {
    "schema_version": 1,
    "cells": 30_000,
    "sample_seed": 0,
    "origin_class": "PRPC",
    "earliest_stage": "9th week post-fertilization stage",
    "minimum_cells_per_gene": 10,
    "minimum_lineage_cells": 100,
    "top_genes": 2_000,
    "principal_components": 50,
    "neighbors": 30,
    "diffmap_components": 15,
    "palantir_components": 10,
    "palantir_waypoints": 1_200,
    "monocle_dimensions": 50,
    "slingshot_components": 10,
    "slingshot_clusters": 8,
    "cytotrace_seed": 14,
    "shuffles": 200,
}
settings_text = json.dumps(BENCHMARK_SETTINGS, sort_keys=True, separators=(",", ":"))
BENCHMARK_SETTINGS_SHA256 = hashlib.sha256(settings_text.encode()).hexdigest()

METHODS = [
    "diffusion pseudotime", "Palantir", "Monocle 3", "Slingshot", "CytoTRACE 2",
]
METHOD_STEMS = {
    "diffusion pseudotime": "diffusion_pseudotime",
    "Palantir": "palantir",
    "Monocle 3": "monocle3",
    "Slingshot": "slingshot",
    "CytoTRACE 2": "cytotrace2",
}

CLASS_ORDER = ["RPC", "RGC", "HC", "Cone", "AC", "BC", "Rod", "MG"]
CLASS_NAMES = {
    "RPC": "retinal progenitor\ncells",
    "RGC": "retinal ganglion\ncells",
    "HC": "horizontal cells",
    "Cone": "cones",
    "AC": "amacrine cells",
    "BC": "bipolar cells",
    "Rod": "rods",
    "MG": "Muller glial\ncells",
}
CLASS_PALETTE = {
    "RPC": "#9A9A9A", "RGC": "#5BA867", "HC": "#8C6FB0", "Cone": "#DAB84C",
    "AC": "#5081BC", "BC": "#3E9E9A", "Rod": "#E6853F", "MG": "#C25E5F",
}
LINEAGE_NAMES = {
    "precursor cell": "retinal progenitor cells",
    "camera-type eye photoreceptor cell": "photoreceptors",
    "retinal ganglion cell": "retinal ganglion cells",
    "amacrine cell": "amacrine cells",
    "retinal bipolar neuron": "bipolar cells",
    "retina horizontal cell": "horizontal cells",
    "neuron associated cell": "Muller glial cells",
}
MAJORCLASS_TO_GROUP = {
    "Cone": "photoreceptors", "Rod": "photoreceptors",
    "PRPC": "retinal progenitor cells", "NRPC": "retinal progenitor cells",
    "RGC": "retinal ganglion cells", "AC": "amacrine cells",
    "BC": "bipolar cells", "HC": "horizontal cells", "MG": "Muller glial cells",
}
PRECURSOR_GROUPS = [
    "photoreceptors", "retinal ganglion cells", "amacrine cells",
    "bipolar cells", "horizontal cells",
]
PRECURSOR_COLOURS = {"precursor": "#4C78C8", "mature": "#E4761B"}

EDGE_ORDER = ["RGC", "Amacrine", "Horizontal", "PRPC", "NRPC"]
S14_EDGE_ORDER = ["RGC", "Horizontal", "PRPC", "NRPC"]
EDGE_LABELS = {
    "RGC": "retinal ganglion", "Amacrine": "amacrine",
    "Horizontal": "horizontal", "PRPC": "primary progenitor",
    "NRPC": "neurogenic progenitor",
}
PROGRAM_COLOURMAPS = {
    "EARLY": mpl.colormaps["viridis"].reversed(),
    "LATE": mpl.colormaps["plasma"],
}
QUARTER_COLOUR_POSITION = {
    "EARLY": {"Q1": 0.0, "Q2": 0.33, "Q3": 0.67, "Q4": 1.0},
    # Interior samples give the four turning-on curves the recognizable
    # purple-magenta-orange-yellow sequence of the plasma colour map.
    "LATE": {"Q1": 0.20, "Q2": 0.40, "Q3": 0.60, "Q4": 0.80},
}


# ## 3. Validate the main Figure 5 cache
# Supplements consume the canonical cell, RNA, and chromatin caches. They refuse
# incomplete or stale inputs rather than silently falling back to legacy tables.

required_main_paths = [
    MAIN_CELL_CACHE_PATH, MAIN_CACHE_MANIFEST_PATH, HARMONIZATION_PATH,
    RNA_CURVE_PATH, SIGNAC_PROFILE_PATH, SIGNAC_EFFECT_PATH, TABIX_EFFECT_PATH,
]
missing_main_paths = [str(path) for path in required_main_paths if not path.exists()]
if missing_main_paths:
    raise FileNotFoundError(f"Missing authoritative Figure 5 inputs: {missing_main_paths}")

main_manifest = json.loads(MAIN_CACHE_MANIFEST_PATH.read_text())
main_cells = pd.read_parquet(MAIN_CELL_CACHE_PATH)
if len(main_cells) != 226_506:
    raise ValueError(f"Expected 226,506 cells, found {len(main_cells):,}.")
if main_cells["cell_id"].duplicated().any():
    raise ValueError("The main HECTOR cache contains duplicate cell barcodes.")

required_cell_columns = {
    "cell_id", "majorclass", "subclass", "development_stage", "donor_id",
    "trajectory_state", "trajectory_target", "relative_position",
    "hector_lineage", "hector_pseudotime", "hector_prediction",
    "hector_prediction_confidence",
}
missing_cell_columns = required_cell_columns.difference(main_cells.columns)
if missing_cell_columns:
    raise ValueError(f"Main cache is missing columns: {sorted(missing_cell_columns)}")
main_cells = main_cells.set_index("cell_id", drop=False)

main_cell_cache_sha256 = hashlib.sha256(MAIN_CELL_CACHE_PATH.read_bytes()).hexdigest()


# ## 4. Optionally refresh the five-tool trajectory benchmark
# Refreshing is isolated from main Figure 5. The validated Python and R workers
# are embedded as compressed source, materialized in temporary storage, and
# removed after their documented environments finish.

BENCHMARK_PYTHON_SOURCE_B85 = """c-rlKYjfL1w%~XE3M^bR1|0~I>^OJB8RnK_%ZY3KD%*1>yBZY?kxda12ypNa&1kf>KVpB*e%pVtzhuv=Umz&S<798u)^*inD6;$X>F)FH)2G8Qe4CZ32%g?=g0fqksk)g2KOH~)@a7$SQ`K`-=jC~@$=hXK23b(#rOM7#kez3F*|fnTzf?ihE}>sC8VRf*Yc85#vsCzf0RLe?0ykK9ZDT>fFf&yYO@Q#vYo!_*ZKTI;f^}WZyP28?XWKwEZFW}V%@Y1>>+BabYpZ%2v{h9!N$_ErH-J*1f?2hKVE|UUtQzG2%gQ-nL_@7q*#-zu{VDjk$}Uv8$j`f4Jx=F!wrG>}_G18_HVt)I)9e;?wHo1{)GwxAnrx+lO|}j4R@zhBCisEIdq$lYMp3mbuIMbAEfp+T0RL*$bVd6$^_hL$EVEk8zYh2xcH;G+F>#&&{2tU@8B}F#=2Mm2ZWP?|4;pU?LRJ(2ub2mOb*YMKO^X4`TjTDu1BjVLB;0Vo%dF^NPO!B&G=0#GcKfI6Ms@S5%~vX!HJ2J6Lu`h$U`BOTUKqFoprmbTc99?=N_DlNjvCtk`5*riC^2iD9i$x^0Lnz37TK99EQWf2V<6BbpYuv*ZIJ=S*m7v1wVHKxQ`IT0hQQh^s{qKe&d--EG6!x!X<2NZH%$KVty`%70Y4Ba6Eax`q+U1oc`Ih{RYQ~4_}>}o-@B}A^L8t%1*-=5wL=Sbu1Ynly0T4YRoT`RubRvEbJYTes`&vn4X7>99jxG;vB_rC3)s4<dxB{+`mm{RrumAb2M)N(%C4E!`MO11Ti7fhX|-+wTJJg&%y8jRoFOTcJkdC4su3|qtq@yi>vt&fOWY?I2nNX7*>VzmG%N5iILl@i%%ly_qbg?#yR{=`PGGd^d>d4ks;+^ExS_Ks?m*dmjsvwoE-f+m=%a`wUU-$(RfR%?<kOiVs%(wuvxvNcji}}*jA|gQY`w%uBQuQ76psMh&y+YI0|4Hps@o+9sRs{^L0pmaArc1J0~D7-qY9R9Ud=k}n6){;2ja&K0JI)Fh;cw^Hi92OlFwAr;MmoIMH_8y5<DpbmDNQKYXD2sARt(Zxgz1VB3Xre$FgyRkW{0GjA0Hrptk{>+I)^MP=<i~ht9xJdMwse3;H9k4cY*24XY2@1E{G{B%WEu()5Fivf3cV*;xe;&(tjI8fC?3S%J(~iwyXpnV>vB2;OD|vQQ0#`VrcuIJ0eve3@oVO3V8Z*YQK$DVTaoOC&H5+2>ioBM_GmKR4Za4XZQ9ZVzC{2Q(w3gNYdqG}47o);wP<I$XtrRkjWQ@p7KkTMG_kBe4hQD_knsaU?-N=%re~*R3!xV}-xOW=xQLxHkk(8gA+e)<^*`1n8`QmHNQ~hejz2YhMC8gM3bC!Ct|iGa%hZxMBt1R;EO+5w0FkFNh<gh-vWr$;<c0<Byu#-vN;n3tB?GS{h-P*;$ttbF?%Xg&Y9v>{tw_BP?XZnYBlSf&+Ck3%sf<wwi(`!BPC}(Gl*@qr>>{@K6*p2_rR^<-uizHOb3GUV;=Q?s1#PL@~5v%c>CRojiEpuGy=szNjvP*%FMb;Ji{u9Y!Vr`#ya5ZT$G~+t@A~@+4r=5&f*x3a1EIJU$A3{DF;xr`xvr@b1ae<KQ6>b_bkaDxjxq^yne9fdR^fB~o*6wmsmPt+KYxKP8Ub1$Jj8sM&V~i<mdCgzypYG}h6gD~bcwOeikM+#w6@BmAk&EEHgWLK)Gsa)1$+88B;JjsWZmS^T?)$-n%8=uWpMqf5=jVql(LR}F1w!}c96!B>aLci(+S(!I#r78VOwsHk9<02$m0z^6q17VEU#<TFLP_PPRD3UVH%R)a=oQ8QNs$Tr-FVvD;*f>BoE=o=<%lATCSj#SNx(MG8Y#6zebmXY-=GzOWsC8h*w^Ew5@ILv)GO^pgl&~n6pYS`{eSo?qZ=HR=-KWJpwnxTz6Xt!%rjcr%6jLtsg4QmY6@hg=zupbRg+iW&%bZ+JRUT**e{icS+RV6Iy=;M-jESUi#waPL3(yYviZId@2(Nxb`Qq5YS>?mm1{64MV;-ON-eOr=DE?|uLR+HBfH2~}yvY~hY^9hOqY}K&UZbiVXD!NtK>@&`3y^%*6mI7v7$R<_bE0Lbo+6P84P6yg5uxbir(+7Yg>%>}xxTET75SGAQAZOYR(8*!1ztelS1{IFRD~P|wnD|!REP%8RP(23cb=9p&U!kdoa*K2$`AaDIR@pCr3LAs}t*Zis=2rp`0SxR-61;<z$L*|0S*lgwnk%D~su62zQ7Y{Tq4>N|Xjo+^MJFIR=YT5Mfr@>svwU3@SuI>md<z?jOd*Q_^IBzKD3rX=V7unORz&jF9W2SS1giWC*v4wY-V%&4&xm9WO4`mfNb@aBp$t4TX_gv?CBY^&Y60~nAj}_M0%Qhq7+TNvV1i%16AN}Pn0IU&-?Q>R4&E2|)3R!J;W9r4cUmRGth->&z^GjYcLGDc@fSg08S_OTnq{|G6k?|Qvp}*InDh*|`ir0#KjDj@fKZxUjKVM+jYwvt>7oOfuF^EfS8Fg<HDjX&9F6pM^*IoJqx9djxzr!cUeIQ`^IQ1S^jbb%Z}sQ0Tj5GHLAlmn0eW+EIzfZXN27Od-h2S-3;{%d6nG(NI!-{LRK=x=#tAYWnE5A1r=y=9KY0dCsmIqrn3o_iL;Unp2fo{L1p?rm0wJ6w%g5O~96f#V^rz$W>HB|kI<Xf@y>z@ri|%xuwHdmh;ppwV<7fLjGEbnZKF>+DH`43L>*JSChx_oJgroOAJptG_VQvgRv11aB-n@Hu{O-l;AK$0XUcB>Kn4N?%-XH(v*_#h9UfnRU!I()ndimn@@sl5K?>87A0lImeK0SW<@;&r@beMuoI(mOhGdUa`KY91^#qs+O>H7~b0DKjG2ZlP@ENE4=2MYzpa*_Ya76f<<=PDfC1NGMTuqZ}#zvmUAVU2o}b%`;3e1{ys;Y=ssU}q&VutGPKY;vO4jC+$c6NThY;pZlpPJ{dJTeZ3$j$_A;26!Fd+$7ck6Ca5o5g=d>!jtu^k)mk-pjVX4ijM3#bTIA(_*V-UZ>x?RLoqiv#+jezW#TzA{vt4xKz*92HGTo`DA=XfhJGY3xM4O0Eo!`5bjQ*3>FrZa)&`jw0FH4Gg)hUA5sr>nov*+N1X~E%tH7`-h@~9302Ft(Zdx#sR<e@-s#F~}m<6QN$>qvg*8SoNNN`4PzYtvlT=3$@7eHz+pS(v>3*Wwb`!pQAdLh)8zCC`I{&@TvsqE;Gz8AWtFAkmX&pipzuuWj&&uo|R;!_J?%D+5+{_+^dd+326C^fpkjYd>!hCt$0gu?45uW<Ztdwrp^xOsluPf+OF!UXEso7ZpN5s&Q$#kRW#gw~gXim(B1(cYTl9QPa23Hu*tNrL~7t+cVmY~k5*_r&WXlWLQLvj>hguq)YkbEbYYASK3!)RXPE94AP-V!y?X@fHVbV4e!hZy@oic1=MrHZd5*UQ8**@Glhrm;*2ySJ5;ku^|Zn@uE5QYz(r_9jXZ;V||Y9kv%#NVN9X+^Q=5CauGn#`Bex14}-v$nF*p>mH?So@fL%{4d8f%bv)I?^R)k=BMgjfqQx{4;|Zz!&=|Woe#kN2kXf;B`})5YI2GR2uMF?bszFGFF}sO-OZfE^kbgls8#lQG-x3#govXPVmsp<(XT}lz0cQw<YZy>zk&u8~V~{o6&c=7hoikAvj6tcIPFPSy7%vM^Cx0K|0<>iX?I*ayPNW80b#Sr^^r~>=LPAmx4>4X@2EJbiS_((0g2#uk$kd0Az5%lZ#Bh!=DcHqs4ZGWw7(pxT@F-ggL_9|iC#&l%d6jhjkN~Ln7<ME5iIP)u9XZ7q<3%B!6&N9cG1fSc6^81QnssCc1J!~f&gfEz79sx^aGoS+TUWoTk{1c%l?^TkbYB<wEa%AEL`*b9&Yd<fG!LXC{GFp#0UlLnRdsRDv|9`@X}o!j(XCgoH?uZzWb*63y!zo7h1TOkG!s6&c}pT%<mn-N_4es|{uyND@rxgS`r*xA-mxVydiLV^^H)#arcd9zdi&<}@#_!x{n29)daTYxXE24Khdf`mU~av9^7_MzcfFp6+;J_cJYhACyKW+z=_*^BsHuhX*C+q=?VA?@h84j*A{Pxycd<ujj1iE<d)0zX!jf(@dh^5k^!bb9m(PfcLz35NJ`X|gI3b#pEg|@lWD|)d{Hy8C`12{sa@bRtdP&Df3#K@=ieW3xOO6oWCnT$bBMHcxQ&VC=mW1nyXf7)NB5{OkKB4Ra;=!QZoK&zFqDBNop;<eJX=yrUO}+2GIvU5)Q`=P7dTl>DEsWJeOTm9U@|zgjiq^;Z!)fD0DY2DtmhrCxYzE5-vp@Ll{ss2{o&*ObCJ;$Rg&gjp8A!VT-^YP-9K|7MvUh)ToJK&V6t+sMvnE1MZ9@LlI5_woe6Hjf_&LvOAgBqg97Wa9XG@*~c~N^<zB(Ku&o3tCxD5r>Cm@M(oq)8<w$aI1G6&am4sIIF=?ocRaT3C);ps`3QU~}nIXxX4C^I~To(5Of<{Rv~$W=k&)pE|nJLfB580co8Q-@QpNxlFsz|4ZIoT&&q#5}DZG14`j?3^3UvbOQ&GxO&&Lxd2{NeE5Vd4(aj{(Rhd3043TL7zxh_^<TxI;o;jyEt7ULfCI|ddm=z#ru<w_b0^rasbEPKUlEKWRCV;ashjVu2A?sB*cQtu^igAU0r%a6CoNgQ^OQ~c!3`x`ZFdmHD&h-#NBv2Vqz2?i2%D9a0=*`f{5`oD0E%}5Rd52!@R^pM%R?T3rB)%q#MC+a`?A=9F7z9DR0^aCQngU^!R8PGkrRQjuxS|xC9%bw<{j*I<%|BbLK%gVj(^yutPJW@7bj+ON<!kMo-ET*N6j+3^iF@%=0>u5XKbG?Bam1O{)v~bL>w3F3tI5=?v-Ca5%<A#zU-(KMb-mEP!FA74quOX*2t`7eu8acgd~_RYuTgJc+MuaMiemAXyjS)h=D+>S`lg)xfeKm4ZbyuU3gD&$KSjBMcpkli9M$fmID!dN9;(zpCjfFC#HAV)GmP2-p`O*Qa#WIUegJiDg>wG0BT+c7m9k(zpE4n^_><p*O<}^&%n*7@+-rg?L`~W=T-y$$N$3AfnL*i9czpbk<x(mY`_?DVn1Bho_)tu-gN_#P+hb_gLY(T7~t_O3kZ$j!^CSZP2V?l_+muAlF0S(WzXwBJUooPx9heS!r8T2NjxZwMM@ey>wt)C!SL36cO@uRL$NaZ*CGiQ?nec=nB&sC|oax^WYfL*%c~59jR#1LSMkzGy;%(Iu87pbO4@M!bixk;lg>*&r=|_UsYW-QDjL8G=D%s4u4$~Rn|Uw2w%aEQ(E0>y^Vy3vIP0Tasx5hntp0#$$FiXDnDPI!SY2JY<Ft@oVsRhjAt(?bew8Hu}~Ene65-i@oT7Uk>HMjManSoAgGxhJ>J`Az5=u?rqUT%7v?(o_H?ArvV|xjq<u&G&CnlJ%%RXC&&p`N(!7GFd(*?dEJTlmz?_iIIY-8fOt0WTV2*<a4}yo|1n4OuKu6>8si9mqw#CT@`^YQS86x|#LwM`?$dl%=D6+P!O5A(g^^*ZB`xKdrqS2rzK-2lLu(1|h08f^U!1MPGLlW;%I525HwZP<fc&XA>eQK>71G#C$0<O{N_20^S0$1}D$sHcU%2uHD(1qe7c^xBDHOd~lW#zWVhkgvv1-2)sIv6-i{#J@YwBizfJ2=ZQu0)H#1NSHD42gqA`1{Fy>)+fzHM02>=F?vVAo+Zb!I1HgYGLumL0z$-IZJ1Q%hDqk4r#e@U$hNDXEOCjxO13%9Dl4&H*S1#fNB`*4P=w*=<#nsKQbCF-Ia^jwGLIe1aQZIZW8=t1hmy9E+Fga*@(Fx0gnfF#L!&kfwLO~d#V#UBvy`b@^Uv&K0ux}S>bc2(_6H7XHjpZ$9LIgk5?CSLJVe#u-YfuWgXmL=uzh`pYai&R(Lt3!=S7KPCG!_0>G<3*|Zh1`5UtSP)<6HIVnLW3TOoiYNT1iW>ETwPJYP2*9XHuc%eN60;Yo^{7>|GwqzzW-F&fY$P0D{C}6lDl*_E%4-5}$0G2h>2x?ZIE3yp?*5mOFNwaLE*|rIaE<;nUXt0y0#k2+_zy3Kqb=~K6Tf=YA;w0q1tiesz0HxB!7<I&D;3g+Hk<-}yEBGonvc1R$p+%d4i3PM;!vCTZ8U=rYF`9`wr3FObxQ4&x>&P8#>?|A%l+W}&?1Ge^KL;vX#CrP1M#F8u>5Oqcj*SK!{p~m06EWykXiw0Z9GE2n<u`S}o*YfIp#gTgNOjO-#F~+y90y>xz@X>ta*AUUxryb6`6#=(fM9(;4R%kTw0DdtlH9E;ipWC6h|}<tsLu(GcdAL<5G8&q2tlJY$l$58mnrbBpH~(BB2B)>6Ihaks^)V_20Nc8;Nx~Pm9-J}NU{dei3~ciyD=w|gCp6lMP9U+l=x5Yl}3cONrQ9ts3%$m-FlEUn;luR(1~G4*o|r<rtp}UV2b}a!Kt1)b0v!(a2($NMANju;f39S#B;p`4l=cX(G$O^N3+b<N_Vhw&kzIqy|nTm1A0h!QJG#mhRBe~OgpCdRX8?osk7?TSo}<1t6mvQP#v;>zEEpAskpM_4nM{{-KvxNFvjAj7U1FbaFQG@u7jQE!S4{9bwlC5Q=pZ?2q_b8#Ewlvf-cmSUMSvI7=~yAxTn&@37mgg8&e`SYI;rwUKO8XZdEInem>$bAA@y1qq|b*ID0Vx%8Mk=4RaXH6(9B;oOB3Q{(xaWQGf$p`>Z=-)|rDmeJjHqV!wCik7e_aHCXZQFj;9|5YFh8hIKr}!uADQ#wR8yWB3`iXxvxd$R@aN`;2vrL6Y+^bvKverG3Y6WE47vY7JI6#F()qlOLSrb&<E20MJC^lfzRJ)M#eh3{#n$2wz6dKhO#ypP|@*ZZy?X31!$w#@?mTQ6-5&0r3{#^pkPJpxoGR7V|tLl$(;n65$~f5qivLU^*h)3+sA0O?M54Rul*Jq}!FrfeI(2-5QpZdTdrmB<si5I=IFNX^?EQC3=%JMVOyOEc#hkU!yOEbG;rEkDIz`!4<O&`Vx&!V!G1i;@uDe-%IB}b{N&(L2dBxjq$WGaXeh{!{K^EVHI{{*VGYO)|4J{51gPF_$OJB{|a7g)n)~Tu>kzJYEv)<rbp_VeuttuKUzmw|LbLbzAUyWSoJv`ebW^&3ADA<n*W?Sr^~iKF$KoIvIKhf0eOsJhl6zn+XKL_vQH_|(e&uhYt$CWdX~9iIXcLO<-44xl%H5n!>q=P>-{0V&ki2KX3oq=Hv=7x9qt~e-v}p5y0QVg9=D%Do=Dt1Lzu)#$`KV1endVft#+_kx9+}bMj1Lh{tUxFgA>uS@64Hy1fOT!y5GA_nzZ4GjBjBslR$7QLz%o*?lAffPBR3sn}p$U6v#f*1dk36Z|vhIZtQd)NCe^Yn;xb2Amato8eBq^uDYVlNgkwhwBW8-O|GydhN@PHn-daw(F48XF(y8zn{2zTat}q5W8z00_FVN79jP+_8^uvD%+b-8=U~G;NJH}$HZ<1U4Xo#z@e+(gLm#jBoPijbEz@<)2Pbp|W_%s+GG;|U;-#VMEEN9Zdn{o;z;Y9BI~gK4qNCSpMGD-NMdKTI+o)cX2Xcez1yj?>9x;bel<#Zpayei2ZS6#xshBqrOoJUcV)IZqk6MHd-5b5@$+d&r#3T*^0bCIjobt^F*m>Ek)vO)4-&wp;aKyd80?O-K-_V+F)+Hb)QsO1-aNhY=Ob+cDx;;R)%vy&`Eu;e&=_ol$4wG*{K33~S987YGV4YVLW_994TFjUP9?`V{?gbw)*#lh&Tr^NtWmbN~tSp-z4usfp%ypIXwPCgu_4Na+Bw@}NF>4AhA+qO^phD$6!C2}7^XdY!EgK1<mWo|0ilORcFo9NrT~8Uk0SFXWnxtE=RgH=R1K$J&Q%g~P<>(S=7mzU4m-KXGR<sj{c)Uy!op->Wsl5%2r;J&Sw+CXr`RknXHT8^?Iqf`bRohe@5EsBmEC44qk$<9)4|Gf9fISL#R~h3nexgl|BCj4h^asB)wEslldWk$JPVwFZB9f-~DNX%U70gPEWPKun1cUzB(oJCTr_+5}f$tGB?1_2&W#dEi=8P(~TKiw11!TVnPy4-x)*QN_4D*^Y!-j!D%O`g#KOp4H;h_OjhiLF7?%ZKXiF|Tt0q^FBX_LyYq1*JX4jGHT6#QCpOnDd^ts{TeC0E=27C$8ITSkthvR$2Vsu-CXC*eY6bYTgl-OYZyAJEmVA}EfZgYSbw=Y0Bb48C}FsS3bvFwE<N`y+Vr8Z9905ZPfI{L%jsm5Rd)DJE^!P0P4+W&Zb0>GrWmYCgc|DMW7CkHJb$AaT-Sz1Al*)M=kEHS*lZu5fo2%PFZCbrdd4evb9xbNp~e8wKF;c4rC2F2M;f$Jep@{n4q3816CW=}H>05s5DIJqg5W4YRd$<V1s)zC2kh6EPkB)`r1X0sdctR|6uH;uN584TCW?DMMdA(4|{6i#Qg47zge|V#;D3NFZFZZ4Yg6_3}v$z_ICW>K<Q__vjHj1Le{$K!3nG%{l>9MJkhi9dl@jK8^;G_lVVa5hsIZSoeY#J6RMk)w6t#*VxYw=xh*=7bzDGG$NM_83skyU{H@X9Y*gcc8qmK%2u6Y&JPkP9k~C!1B9`h&w>+t_P|4uxi2ao72*Pwy_=Q%_byj0UTV))cx@+dlhK|mMo~|oZf6D9oe`}mh0BRz>Cf?aviF>?&hSpl`z^+Cjz8tCmrE@7Y^)g=4kuPxa^BGjJHdJnv*g3rKV`uDq?>a$Szr)FpEzV^V*86FIza0wGQ-_#0uBCKf~~1)H&M-}>mrcVUQ>*4uNTX?8(b6tv->#1=iTv>XRnS$3LzI}V3_G5U3{A;HU-0UsT7^Mwt!hvaSSxTj_4EKvdLPp-@Qs36lX<ht*Ax>AtE<^Bz29F(VA5yv3I(Wvc#jHvT^G-asI_B8CmbpYGM>Pj@ZIXMlio}aHqp1UQ!L|Obl2D44-<6jYXaA+i}7(rF6)nL%Mbu3y!Jxll+3t-_bXFU}2dkbocf398vh@>#PP2Qu>~zo9(#khJ0rDMp6@|ZkqnX^gVjh6jmSlwj>Ww+|1sLNH97-n^=_gBfac@Xn4X6jorYZV$Xj>a1t{9J&vNhLMZ)u7?gLZuL1r(_W?RLTz0cBe1|F=I#I_hM8n+wVYpHT6`sSK6|6C}w{C_b|AoJFEsmHAc;HQ>IalUE{%obF2VRJm^e$vju)YCvN^}8d>rGj(5;8Y5Oa%@~gc1MH;ABg9a5b8eE_vG$9a<Ba1FA*fU(5dNjC6G6eGs3Skgjju)1vP`9WyuL{gWTtKGByvoWERC+qOK}y#`OwV|O@xjou&4^aL;LC>+Q)=UkhejB}q<hVron70+#lX(!H9y#OX%0b#k7%XzI9H;xlSL1z&zdfkk$={Y76YS+^@OhOBH6|VA<^Ndn1rx{Ly{vEsP-T{<UNkX#!jDvQDLEBd0Nwu&$OQ=XR6{CEsB!~>4srdvdn-UtbyV@SQ$jww=e$^Kpc25Eg#4Ejy(iv7uP5^q}@ADNM<P0s_bAvv#Jx2=H@u-(9=0+hu@95E&0KV$|TPbIayJF(^%A4Q|sB5Xq%9SsF7u5u+cy6jr5CcS{liF@ws1tYhW1y6#J%NopJ@s-bC^OPYsn}Z?G<Jfxt&$*&Xs%<j07*fB4zUFTmUQ=(dxH|{$|AquKs=QY(?ti*RDneuFqp_PxiNVhpr)|6TMgU+pC>vixyf0eWx9-CgW0&c-UE!fLRTvr8DA67!iKi|Ef>etH{7dB9&D4<^IbLd`zZu*!S*OY=h{)<&+?t{5Voa9-yGW}vmRBve>M?*94b}xS(S=DikPFKu|4JGAu_N&bkSQckNsm<U78i7&;+&?5y#m;)8ORPC>_4*Yp?mzy-0nm;206Z2M%7s!l>sZd#4VQfzTyIcWBWk63Wx&7u^~LY1MOou?$9Dx?!c>I`Vx|-vBVl98U79S;~D>W=#*o4^<{D&I327;!-Kaf5v`ph|QYCWlzYvn@5zD<5DvHdIQ}erP{jC6<)e(F7zd>j@&6@3!qQmP(eo(^(-exr3}Ut{uu0bCe*;w2UksBvN;_}+G8xpiqba*<Y0dYAox$G92gx_7eANJ%~+#K2O<vnhy?TOaYSl0zP1Y!dUr?}p;$_;u;izL4@UHp!ijd>kvF@wnJ1h(<|!wU;;(Ird!q_IOA*m-1bS7>MeY^~$SiQn$jE9jnwEcaV~0xd?!2589ar1s47kS6HZ)b4NhA59m|P7xB5Uy$vbO8NZK=Wda!vTyh=?IQ>0Y^XM2IPrcUDy3ZW;TPivFE#qrQzq_u*81spf1WPF%r(n@e!U!Jqx5vbC~dgea8rp|B@ec0`1(2RLQ)diQPUoNo6(fUnum$&m~?HM1<I7RN!GQ%0PIe+ncz3}SM29b93_pa;Pv$W04@tmZ$2ce7o|ONar2uAPKY1dtOh>bKs)_J*g7?&R>Hn`$gtf$n$uvPqW4%0x{<k*w{2uo1H0K4T(5&<Od=UF(yOh?j)4Bn6>(prKCj9EFiG7>&n%@0H3*TSAOrkO+?G+F@2kCj!a`bLzusw1|e!B3@PBBcS_@T6(KQt=8@K2H8z!h9+CC*E<ic9J0Q4Gju~8y?lv@HuP#i-Lz=Im8xs8%nUJnY{~#rIAV^GaMHJ28`C&A?yXiUAGn}9xp5%kHo4Rq>lLPz_$Wv+WP-#jhN6qsoWj-s6Jjb<GgB0-cjLB7`x#ClxZXYFp3A|v^FX9)^yA=4B_-<#<DYw3k?Q0mv_gfDaS7=w*=0lp?cwq~>iJKENlcfXvKB8BPp`^J5ME8bN**q*!-Rk6QO)((#Q`vPpA(2}4OOgJZQqHQgx1k72UyFsh$aueX+T`S{FwShSx&xxcx`gwaixNR{29Jw3rh!m^_AN(%;(YKI_!ISJKB^<1a=?BPV`D*5}s`&xY%v_Q%`?OF;`5B=mYxRfdf+Lp@!8%Oq@@La@g}kCGNg`Frd9ZP_1XGl{Rs1v1!TRy^X-t(R8IX%2)7Wi%$&WHlX@Z!Cz9TC+VSYH)4d9Kyki?eXk>%)m77oN5!E_4RA&Uk@AwJ44S#1C?@DmdIdzldFNBMC^GZn3!Vp7Nh;3^JsD0eAzzSB&P?bRW0yFBRPiAX*3`t-9wc=K0#y>zl!OrzGR84j__$wlfjE7i-@Y|L$0NFi^gx|yuGz>XqjBL%_pKAU7|_+1`K8FQTu;~lELCpaZ;(oNazlU)EB?a>a*QCtkhibt=w>5<x7e97Q2X(6aD&zwGpt59Q0=CVz->kqAvYBi*}}p>c2CZ4uuS$_CwOQ+Y@6UOdo7bMw@&m`IVM^|Fz$p>AUArfqdT+P;S}+@X*--Pnd=RqnvTQA&x5-{lhhR&&{VR^Hs7;o^OJJ#dJWpFuE{l+T;Mp0^`5hHvy8?{8}Duum$j#D!bLHC_BKt~;JH-()|y?V{8bBP2hM=axVh10+}zP-B!T8xjL2!e6|~Vt*8cjs{kMX`I~ZXb)b6Z;M2B5%5?wR3l(ZtxQ1dMSXziG7T;0%9@f)<;Z~&BDdkr`&_}1Ga^Hu28w{i?TcXIl2>o2e&o1_N|c36LhrburcWcu1zwib>1LW_N?3sIi*S0UQmZuuB<ZnI+r`6z&;MuW12w&&lCC=6S1cN~9(?@-8U86%NB2DE?njSuTKJP;dmy1eaehMaj9i%$u-oMcf|7XbywbKVpR5@EzBr(&oSm=ROXMChpv25WR--E37!P(>=<E&`|aKfhK$;T$f$2r_T0?Ge}4l5(*60XTWChMr^K^biWcQw34ZSjW^aehfh;h1mH~wlCE%O@hB7T=+sBLIS6d-`3&QSkhBi)G}jJiN(o1!IYz-s`z;c8n{~EP1(#Y0Y<dtDH#qbiNeuu9{rwn6j>o;Nu6nBc=q|W(`Zm5o|!yLv7T6DNM<$;dE0rLtpmY$G5~(tj`~~wa&He)Ac=O1g7I#Fq<UF-MJOEY3x0l!seZrQP1oYczu2@whu2R>H}%0`bU_8eF6J4!Wg4sy2_nSU&ul%~Dd7}FH;SD&h;EaNyMZ+rmkpG$Q+y%K@a8{#(lT2A2~F{Uo>raryzl;MBAVO~R(4!sJs!!jo}e=Zd*kWFo-gd-dEIxM31b#wn3rGh*&J0=Gx*6E!GplX)`ug*<LgMq{lon5Mk4^@n1BF#Gqkfe6wec%r{h{7mRfmrPilPPP1UQ8(>|k1<^u{-^3w0G4@q2a$yQfK*VN}YE}Z9Jd7nI>DAcMEq{3lrh+=%VY=FoKR{(&c9xb_foacBKTNVM?QqnGl0yu>5<mjlmHd1yH=)jB|j7Vq(ppM7aLl>b6i2+gvXqK$9PZ6~p6xLVz(>1Qkz`ZICJ|iR)i@Nn#cxsXyEw1TJrYpS#*D_fXm3)f&fU*_y^U~Oc7#1m=s27#Ns^}^XCP~$l4jddD=n~mnoSPw186zx;y`|G_^Xe%FG2|I6It#?=@;JQQvO!jHxJ1Bp`rRMQQ#26?Cn5vp4C-NB@hvy6Wzn-gjdQWac`%I5`00(1jyRGZpZfs}49PPyMa!#pH)*p^HV^v;XfzjnF1bT3KBX|*_U&y$DTYdPyJ1S{Nr67S?9hs7e>Y-b+s?d4Ghp2P^@c}9b%_ai(~F+#bCUfy2Z>X7o66*2n3qhc#1%a`C8-xHdRvlT?ZVs&DzifmUfNVqia()Q1}Oc~@k?z6w}~4wv@c9a&1HN}1r|bQIWZDnQ?yR8>(ezNeck!e)52u5@++&^me&ttSn`;SJxLmyRal0q$k)lP6>Cg8^R_0tCb9SQ96MmLh7TT)Wi4l7`kAlZ%dBA1Gd1@!Tj>2iedd{!oaiVkY+m!4Qw>gCD@2OhXSEJE>RV{^vm{zNFS6M18eNB@t6lQaYfh95R;^pJUtYtbgm}9Dio1xo<8?WHIzWdv!6~fVqg+0BwGWq3uC}Jcd4R&M2p%3j{Py7RKOY=Do|sq6bU>0aK+m=V2|#nKX_vY0G-yi8&hR}@CnuHObQKREGxOQVV~<>tRGb={0~?Y<md$1zy+WXa*!icB-<j1M0~6BWYV+)_?7b}08Fa%O=()LqC>UZbxrzih%Sf(TQqd2YeE@(3vQyH}5S8pazgHogaUkjS2h1V*m8rJIgK_l<54qWek%`l@9_%*haIaLy-X_7T|NXz5r!i`IIgo*x=d*d*CP3_WGI$L*IlIqnYhOnnOuTu1qe}!vd<jI}mTKF5o>#=k>YNUzME`EOPo_(7k)IxMoCki!VrnfebrinUlf=u4p_%fYyBa-Ec=s|o;fHac_iOzBQ--5=oZl3{l@G^5Nl1V>*b5P&I-QZd&NM#VpP+Js@kTtTyLw0YZsH}T<JfH)LhD38AV%RS-KKM<bZTn!kf#?i#`a8cwC5Cnu-ktIKp$?1>8`0(E}xgniMrr7ow{3RtxnN2`#Z_boc50w)>uN<i5Sz-g2Ibz*#3rK@=g%h3F~tiBixUvtc1aNst1h)=czN=aPk2yzd!iB;8|T!B|1EGWz9nse4SU*E7j1ExhP+<$h6)?3!toeAni*ZAkpZ-S$bR-6$n}d#NQkQqBn^QG<;%{q`H4sOfKs2R-ABto$SbIf>+PWte38_rlJpZOGT0J!=7#D8<f5~zq+xn?EX;4dn~iWWpC^&*QhbrVzDoL@U>DdtVO0dnw)jq1Bkkn9`5;=VWul)mtLxm&~4ioMb8$~CMzW*MY*IvPtDGqJNGE(C^-xcJc$|sajl%dNbq$)F^{!&j#sxPcVvTsdOyLzUHrjBhwSzkyn(@qq5EpGn|uOoRPGv-ye;yMv%bDxY}smbZACAZ8$NZe#SEK!-qWdm?`$}_zE4E&=28r&QV?_Rq*9PGWQxd(VIpW*JB`f(a_+?Vq#T${g+yAFUaG~gEa~N+O2S3P^k%#qbdCZ6Yk$n?w}-ua3P9d?rW>rb1cm?$4Z#+S*aEl8n}pcl)UGS3jq8d5(5N1K7>tY}dFmbC7N;a~*g~R^c;}Vc9#HOwtVhBMPx2At?KlbkN}o_xf}O|orY+rC5#^Dn=mF$ykct&zsL@i0L<KAJ?_H%63gi)ABTn^~J)k7hReit<$gy$29L)M~&wT~q!TZdE_wnO?9=vZ4BJf@?Cjq8%=rG{5KCASj^{S;z&jaixGBYda**$q@5Gl`I!NURn)jfy^>^i#7;5jdk$+K-FfHgOmZ!tuC(Zt5iZ4|~yv<{>lYf0$Z=cj4EP4N%D@q;A8HBjh1W-@q0XR1$J=DCsguX$D!+TkMEI`%?h$~a@Wd2_S1Ryg#ci_#dvA?M3p?9|VBbN+`;NkcCjJ~kDQDV}dM7jmAVh+J|)WM1iHSYXc!^mpaGhJ>9S8^0}^4JXmGu@)#2FftzU9ytp#@}mi3e4OY~TYCFt@#t{??dW#woZG}1=LQ8uCcsu&VIh56g?lhfv$b0g^Wht5V11#4cU9R!1`Z|Cp;OSOEv+*E2R^nV?i`@xy*L@_(R9c!kG2$AYpABq8lzzJss<IzYGfJqg0vd|TyAI&<1ac5u`zdP#Qmk_n{2xm=D~q+q5BAXHFVrXl=HYb?=ZdKq<qJolDte<UoPc1#1Wf!Avf-U9Wz0L$cw0!&?eG>FCZj~oT8&%=GT(%`Ji*}gj*UNO>tsljlwG`!j6tOX_h&e%yyiTWdEY8X&9h0EZZ|N;ZNNGw*~8MG2h5l@Wvx|vVn>0E{QiAd`lG832z{=Na4zZ9qAD}=iO)NIfr$|2@f1&eS!&h=g!S#&ZQ%i#7L;RCX)wKCEh@$0I>{wCnv|4OlI!dV}rXLYP$Ajyyu8By~&ZkYgzl>=Cw2ZU=PVSIp##`0;_LGHNOTqloy;{Rb{0?C10$y?*P;f`mw@ZX+d3WFxW-a&Glh4Q-Movc&<!U%^Y~NAQ6|wK~_Z*h$5IS#lx4~N*ARvr^H<7{%4%NRGOcl;}AW@Y6M)7!2jLUj2WNyZzl0M8Xsfg-IiX;(e4!-ezOv|{gaMHRpo(6C6Oh^cWH9bX165thv+o(0`eXqR(os<=MI!nJ-xREkmVyid#Bp}*snk^P^)$@RHKKP_HQ`pAM_C{xYH@g>uI2`8`+D-Dd*I}M!@9I=MDZC>^9<>lD5(4F87i`KKQv*sa6!>>3F!MMmf~U{`j`rf7gh^0+rinJm?tU2e&0(Skehi{@o-{0vKFWbGiMATRL#0C%4+PvCi+w^sg}aVJbbolSzOhN@5qEY&-ZbYQFc|D`-s32C5nR7uIe&GY#$SXFqmlZ$CSim3OlO2ss-K+n3I9BIustdGtlJp0$H%9zh_HN&;&;2_d<+ug4>|mVgQ7O@sX*Nb`^s>?9j{Vu2Bt>XSoH;BB(vA~>(;4f4bt(6<=*sV`>_4|Yg(7j?-`p9VLuSwA?;tTouja9zJo)~2LhiGuh+j_71GSXbyO#Fjt{jDT0(#1Py~Q@eHuZWuD7`OiZZ<;WM8r*xk?iVkFsmJn$)hwLQH#ES%t6Gv)N#X>Bj07g@;BTqC6mO?mUot;Z!jRK?;9@9vRYp-0pEFMZ%g6fd%&RtT4?`G;nxl<2Wq@H`QLxKLQFu_;Bela^d7@ig*cJgS-c<^Zlnf@NPq~}(VL#-23b^`JAwz0=laj-kf|1S<xwDB{1nBun?It17`*|_Zj!|(-xJ6~VmGYrhDr>y(Co?i@g_<wqKv2!KiH$J-P=oK9|fL$vNDQd8{UqtZ3;hl+j({qTyAXYn}Z!XDM)!K^uvhX)Ux;+Mtyy*qlZb?S}O~T+4FHUGty?Zo!Ur77W7lyQTEa9ICRUcxgn&I3Oul8Hyx<S}qW}6;$rrHf=$M}H>vS-QVmrR0oQ^nr(cxMiMA@YTHpqzU~a+-S!KWDqwezdj6cYStju_1OlyA{fgsu=>~sy|t(N8iHcmD4Ldxykq6#@CKtca`zg)1jN4{TScXnOsx->|ppHGdkgLN~RfzfOBC36GmEfKxb?U*#vr|7W$z^ADaAqzWjujM?N!){PUlcF^?PTh>H2VYw42Py7vKBH=6)W0QCv)PdZg(>M;7-{Ht;Fual<zM>+pTJ^u&3dXJYHLwv5ncfI#NbnV>#j;VJ8Q~+xwEQ>X@+Dg8n+&=tj=S|A1`{q<y4zupl&dl#mzCZew{>BZFibYt@2flyYT=(S9zh5O7_Z;rNTMR}f@b6j)=F$te>{f!YR5nxxMr%3K$^xz~*x)o#i_5_R)3qm%E>Efpd_MVH;fEo~vnVpm=nnZ6DZ0$Y4@bdoe?B+}_@M}@j!5O3!qH9JvABO#UDyj<`X-MMupzK|^8Hswk5R{5;rLdZtTH5JUz*TUhTzCh0pjFIeNJh@Z}G2M7qFvu1=CD^i$W~I7{>C}cQfT8>KUasgf@s0JB7&aT8}pL-gXJ;sF5B8B=_jfP;~k(U1QT{1KKm;Sl(ShkqFk6dUjUg;DBm8qESX+h-pY6$g~A5DvSb+snqs5p(V!7Ix0;;*_0qqDL}o9lR!Rs>pAh?lp36Xv*Hu~9gNsp+exo7=(Uc>A$AEB+0089zlrz6s(g83TAxPrk-ild8OVI%l<niKoD3EJvHLsm$_lup`pk|_Om4Qfw-cM^>~^#JJsCJ#j`j9VSkqp+6GEne1xcx@Uz(C1l%~i+X&O%CMkDgm=>Gy~tNZB"""
BENCHMARK_R_SOURCE_B85 = """c-qBT>uwvz75>ksm{K*eTddZS4Hb?X2T*0z4PeWHB_}_EfEex$$%&RT+nHI>N)ezB(I@Pa^gCzvMv;o$!fp-QBIk1ET)%VC!GU;VjV&rAo(gG7aaC!v^eY=Op=7ZVt}#M-p@SE8V?=3-X01$+L0cngr+ne`T={%*aDcz!&o-=t57Gs(aCR-iN(m=7qOgr25({1_v1CM{s><`DrKM%<=n0|%y0x=~@*d3O;(Dd-gkMRggf-P2lNc`3H*4YweYq2KFxaXjQl+|8m92^A$QLe!j$1JjUN9lA{?;f{D4|U+b8zx!RRrtsa)TdQR?p+@@gJi$Gh$&@${=T6L4ku9v!e8gY1Xq+uf+>-a+GGf6r3uXLeUXrJgGN%|K{{E6PuNWAy{u)+MoL1tXV=Kjaw=|-AA*i8XuHn&U_{+C7qcqovc@xu&(TO5@&6QAHiOp=V`u6>(sy{T&0y0x5_bxY)|dQLPLlCM%T3}?}?p3m83pZ?oyPk%%$+PgwjE-Q#x68h5xWOa!@k#;Zhm3ORKIxVFjy$y|oNZf5^oJyb&Hm#@tn(m!_Z(9(ywE(H*hC?2w*_44k+sg|R`*6}$*<%p1rBUt{uX6;`$!e5((o2Yl;RFO3R8Jr|W-9vbV`veI6`Z**}ZZY3OTUhR>T<cC@>&rY+>z8O7WV1^)y#@#C4iM*}fjSj0_ddwSz2Fn-;3g6T<OkL)2E2nh<OLsa1O-Y(l5Bi+AF&sUXw8cT{^irElcnk4~7RPaWm@J3)_j6Y}y_N_+r@*7E@ZCZQ8nGyfmGzKZGZ@u~{EQ9Mg2FDj_8mSDz1fGujZ!xeKL%d|e-soXXJ82G?M0XfuQ1{XF=h%~%W97n?79XAgnFcQ_%6BgqN&KsTd5H^z{OI`@5C$F0PUm^YyBZKZV%^cTY^LIo<*PzPN^X9S5~<Q5Cg0}C3PPk_4M;m(C$*)8^w($I!B_1p@_0izV3+vk-*3<IokxUOBKT@;<{5CaeN}cwUWyrOPo_?**FcZ0XC13MVcx+uBA~`@wf{&|M(jt*XrB6@V6Nw{H?4SI0L+eY?X<XUalzGl{BsEEE*SRNmc;HG4^*azxUvyMeo9zQgVVjj36)GQ8ufQy7UqzIB7YW^&I8z#_}GBx$WZVzoO*_fH+_QV|rXqCKP?pihqNDq`2ho69FyG9ikw4!|0ZdR@|ywoQxN)K)6z8(9-z_=!;Al#{0gIu;liNlV-fIj*;+m>rlsVI{7u7OrU8czBz=eII>jILf9|rQifIHn|U%lJS3-jG0ntunt{?1K{{SVl&s+?r*7$!zns3lI#09K%rC_CyZ7hmkH<fyX*%KTU(#cAGq33ZdLDH1oTx`1qEAuL^eJANrW1JPe|WVw6K+Giyl@H{OF$?C3ISuizU_DfqOI)a2V_>dfW;`tWFjFIy*m~XzM)BrZ>f-JWt_wekCS>Kl1iB+SftBoDwu|<JdiS(b~a(*I$MB4^1(jad^Kbi3TD9U#tuL=#w7~Ty|Bc||60W?tPo{^Ybwc?QPv?#Jx5;908DM#n5R0=EY)6I;^fM-jBS=wAiAa?lmq@b0kx?f+KWnhFrCLrl3qprA!ut8S|Mvn+C*@!3V^2KC`2CM*EhhPQR1aS6VhD0{_f({?DFi2bbNf&em{Ht{_6Vt9WOqcym@=^_U!ff?B#cFc;)11axuGpdpUb`esRt#PmYd`Sk;H0?+=k<vA90YD5r9Ir<B0u*ynz4vW}3F!0lz92}cDvh0C1XmJSg3CPUoBh@$wId)h+Hi7dIvDg4Rvj}g%;RowIoZ74*UK0?#Yw5b~oL_Xg>FcCX{KIO^P$#%%^dnRf<U!6=DVQNM*>2r<xW~p4VIw5}XBZ5h-rhuZhmBC4r{K>Rjo<%^O-se<<nFokzQ|f|B_8sx+-@H@@x984>cnAFjFss<QL(RyZ8!x#HVUf}aw8^?lqGT!Csas=9kkHk`XW$Ge`DtG~edX`f+Eht#$uuszL_l5=b8Y@K1}`GCU2=}g#E;gLOW?BUJ47PZR)~E&Dgz(}96nrBQuU12;HeYlgd3yIdtU>s{66;`j^^P5P?=KK=sy1!?xK$kgkYH8q}5*=5$!_Q6RL+rG8N*{2siLKM)GdxEGR>V#K3f&B{NuHnL`sd@Q7))FW|acaDoSKrRkS|#@?eoI}B2$9rCGoq|Kw)Xs3c|Ei%V8esJpVZ$xau!5&U~RvE;HHi=J#BO-Qn=X3J;1iC`WSau+?JvT@uv3=l#5I@s4x=Tq%pl`%m+#r*WH|R&ca8!%&79t96L_vzp`(eHhSRX{$`<K&nkl@~!gM0pA)-lejqf7UP9q}5k5@O6^GfmD#Ly+m{=GI8#6RyE?-lD+Zp@)!#vmTXC92LblL6a04#!@XLD#5KWiI_LK!oxCB^3-FZ9vKgX+7bxO2ENqPM>FPa@hN_8O4RX@NP|c+hG!yKTZDNo*cm5dEAMFfru;DQ8VCdEJk1ZZc^>I8u4*5VjoYs{vq05!oc;0mm`b~+M;XeO(b%j~iy)pn%bq^_8W$rWZ5A2`co+L!S%G~60KZk^w%@6>D(@JWZ5$Ka(IAI7M~%i>h}3$HyAc*OYM$`elDN23+Fu}QKYOUHrpyWmqncFw>mUC_(@R6YMJs3Hq8@pg*H1V++l;#Y3O<+*o-#6H^b=Hzvq6u2snA~-PsE~QjmKgGD0BSQ+*8zEHKpaK#*Hx|Y@_p9n<miM=26)ip*0b&G=}X>^PcJcf~PTMGRpC#UM%?00aG%d3`8{K8t{KS$n}Ld-W%z@Str`(Y;Tp+flkue@3t2?&bd?Ps%-=?3`&|pEdm2hXTzv$s~}<OTRae=ZdZ=IDfXyu<18ZX5fYl1OXPBTo+uD^zkZ3d)0ZP(ChZKZ+b8Fp{>pCR8BOflL4Du;4fFX2vyd;iAlWxCfWLWpN5-kN4=wWJTVDd89(dVjq;w!w+2sXFhBH2wV3GDpRnQ(!v0foLeJ@?NwYpu>)>4IQTmSX@S3bdEL^KGGTCmXrt}&Nqr*xg1c8;VugD>GYDB<3y#eE_A-jD9Pa>zX;ir`3M3S!f!4Ha^)RGA#n>6<m3zG2u(93r(K1at${u7kvz@pm3C#1mBAh!%(1h|2QgBc@bTZD)ZPcM`zwlAHDK+YG^iMQmxST+VVoqUI`46YiFRtoayqq`~y32T-Tn+xABY>SuZMixNh176737$nS-qUXR(1@LSBMQIRZg@3m2T@xW_2SB$4cm5nR?7x(Be#LnQ1vvz#gQ@k87*H;Ws{ni;xzklG7h3Czo0N9=$KIi1I;gd*eWMg5GyLh>qSJK3TN`K0&-cwYSkzTM}PQ%uC@<QFoM&9u-#a5N9Inz;4;^D(ikRfSsx&ltN{nS;nCM4aG`VV3Q|B|^|_wyo(CNsXG7c?RwKsWN<eZ(o!!X{1ol+_+465rAuJ^z>}X5!*>hD21&SZ^OsWSyQyS&e@UQ~r;V7<VK);_jxlPsw6iA@%>sWd8;Zaw01"""

if REFRESH_EXPENSIVE_RESULTS:
    if not ATLAS_PATH.exists():
        raise FileNotFoundError(f"Missing atlas: {ATLAS_PATH}")
    if not TRAJPY_PYTHON.exists() or not TRAJ_RSCRIPT.exists():
        raise RuntimeError(
            "The trajectory-tool environment is required (environments/traj.yml). "
            f"Looked for {TRAJPY_PYTHON} and {TRAJ_RSCRIPT}; set FIGURE5_TRAJPY_PYTHON "
            "and FIGURE5_TRAJ_RSCRIPT to point somewhere else.")

    expected_agreement = None
    expected_precursor = None
    if LINEAGE_AGREEMENT_PATH.exists():
        expected_agreement = pd.read_csv(LINEAGE_AGREEMENT_PATH)
    if PRECURSOR_ORDER_PATH.exists():
        expected_precursor = pd.read_csv(PRECURSOR_ORDER_PATH)

    refresh_log_lines = []
    with tempfile.TemporaryDirectory(prefix="figure5_supplementary_refresh_") as temporary_name:
        temporary_directory = Path(temporary_name)
        staged_figure5 = temporary_directory / "figure5_trajectory"
        staged_scripts = staged_figure5 / "scripts"
        staged_result = staged_figure5 / "result"
        staged_scripts.mkdir(parents=True)
        staged_result.mkdir(parents=True)
        (staged_figure5 / "input").symlink_to(INPUT_DIRECTORY, target_is_directory=True)

        worker_python = staged_scripts / "panelc_orderings.py"
        worker_r = staged_scripts / "panelc_orderings.R"
        worker_python.write_bytes(zlib.decompress(base64.b85decode(BENCHMARK_PYTHON_SOURCE_B85)))
        worker_r.write_bytes(zlib.decompress(base64.b85decode(BENCHMARK_R_SOURCE_B85)))

        worker_environment = os.environ.copy()
        worker_environment["MPLCONFIGDIR"] = str(temporary_directory / "matplotlib")
        worker_commands = [
            [str(TRAJPY_PYTHON), str(worker_python), "prepare", "--force"],
            [str(TRAJPY_PYTHON), str(worker_python), "orderings", "--force"],
            [str(TRAJPY_PYTHON), str(worker_python), "cytotrace", "--force"],
            [str(TRAJ_RSCRIPT), str(worker_r)],
        ]
        for worker_command in worker_commands:
            refresh_log_lines.append("COMMAND " + " ".join(worker_command))
            completed_worker = subprocess.run(
                worker_command, cwd=staged_figure5, env=worker_environment,
                text=True, capture_output=True,
            )
            refresh_log_lines.append(completed_worker.stdout)
            refresh_log_lines.append(completed_worker.stderr)
            if completed_worker.returncode != 0:
                SUPPLEMENTARY_REFRESH_LOG_PATH.write_text("\n".join(refresh_log_lines))
                raise RuntimeError(
                    f"Supplementary refresh failed with exit code {completed_worker.returncode}: "
                    f"{' '.join(worker_command)}"
                )

        staged_panelc = staged_result / "panelc"
        staged_shared = staged_panelc / "shared"
        staged_orderings = staged_panelc / "orderings"
        refreshed_metadata = pd.read_csv(staged_shared / "obs.csv", index_col=0)
        refreshed_metadata.index = refreshed_metadata.index.astype(str)
        refreshed_metadata.index.name = "cell_id"

        missing_hector_barcodes = refreshed_metadata.index.difference(main_cells.index)
        if len(missing_hector_barcodes):
            raise ValueError(
                f"The benchmark draw contains {len(missing_hector_barcodes)} cells absent "
                "from the main HECTOR cache."
            )
        refreshed_hector = main_cells.reindex(refreshed_metadata.index)[
            ["hector_lineage", "hector_pseudotime"]
        ]
        refreshed_metadata = refreshed_metadata.join(refreshed_hector)

        refreshed_orderings = pd.DataFrame(index=refreshed_metadata.index)
        refreshed_orderings["HECTOR"] = refreshed_metadata["hector_pseudotime"]
        for method_name, method_stem in METHOD_STEMS.items():
            method_table = pd.read_csv(
                staged_orderings / f"{method_stem}.csv", index_col=0
            )
            method_table.index = method_table.index.astype(str)
            method_values = pd.to_numeric(method_table["ordering"], errors="coerce")
            if method_name == "CytoTRACE 2":
                method_values = -method_values
            refreshed_orderings[method_name] = method_values.reindex(refreshed_orderings.index)

        refreshed_agreement_rows = []
        combined_refresh = refreshed_metadata.join(refreshed_orderings)
        for lineage_name, lineage_cells in combined_refresh.groupby(
            "hector_lineage", observed=True
        ):
            lineage_cells = lineage_cells[lineage_cells["HECTOR"].notna()]
            if len(lineage_cells) < BENCHMARK_SETTINGS["minimum_lineage_cells"]:
                continue
            agreement_row = {"lineage": str(lineage_name), "n": int(len(lineage_cells))}
            class_fraction = lineage_cells["majorclass"].value_counts(normalize=True)
            agreement_row["classes"] = "; ".join(
                f"{name} {100 * value:.0f}%" for name, value in class_fraction.head(3).items()
            )
            for method_name in METHODS:
                finite_cells = lineage_cells[["HECTOR", method_name]].dropna()
                agreement_row[method_name] = float(
                    spearmanr(finite_cells["HECTOR"], finite_cells[method_name]).statistic
                )
            method_values = np.array([agreement_row[name] for name in METHODS])
            agreement_row["agreeing"] = int((method_values > 0).sum())
            agreement_row["mean_agreement"] = float(method_values.mean())
            refreshed_agreement_rows.append(agreement_row)
        refreshed_agreement = pd.DataFrame(refreshed_agreement_rows)

        refreshed_precursor_cells = combined_refresh.copy()
        refreshed_precursor_cells["group"] = refreshed_precursor_cells[
            "majorclass"
        ].astype(str).map(MAJORCLASS_TO_GROUP)
        refreshed_subclass = refreshed_precursor_cells["subclass"].astype(str)
        refreshed_precursor_cells["call"] = np.where(
            refreshed_subclass.str.contains("Precursor", case=False, na=False), "precursor",
            np.where(refreshed_subclass.isin(["PRPC", "NRPC"]), "progenitor", "mature"),
        )
        refreshed_precursor_rows = []
        for group_name in PRECURSOR_GROUPS:
            group_cells = refreshed_precursor_cells[
                refreshed_precursor_cells["group"].eq(group_name)
                & ~refreshed_precursor_cells["call"].eq("progenitor")
            ]
            precursor_cells = group_cells[group_cells["call"].eq("precursor")]
            mature_cells = group_cells[group_cells["call"].eq("mature")]
            precursor_row = {
                "cell_class": group_name,
                "precursor": int(len(precursor_cells)),
                "mature": int(len(mature_cells)),
            }
            for method_name in ["HECTOR"] + METHODS:
                finite_cells = group_cells[[method_name, "call"]].dropna()
                precursor_values = finite_cells.loc[
                    finite_cells["call"].eq("precursor"), method_name
                ]
                mature_values = finite_cells.loc[
                    finite_cells["call"].eq("mature"), method_name
                ]
                precursor_row[method_name] = float(
                    1.0 - mannwhitneyu(precursor_values, mature_values).statistic
                    / (len(precursor_values) * len(mature_values))
                )
            refreshed_precursor_rows.append(precursor_row)
        refreshed_precursor = pd.DataFrame(refreshed_precursor_rows)

        if len(refreshed_agreement) != 7 or len(refreshed_precursor) != 5:
            raise ValueError("The refreshed benchmark did not reproduce seven lineages and five classes.")
        if expected_agreement is not None:
            expected_values = expected_agreement.set_index("lineage")[METHODS]
            observed_values = refreshed_agreement.set_index("lineage")[METHODS]
            observed_values = observed_values.reindex(expected_values.index)
            maximum_delta = np.nanmax(np.abs(observed_values - expected_values).to_numpy())
            rounded_equal = np.array_equal(
                np.round(observed_values.to_numpy(), 2),
                np.round(expected_values.to_numpy(), 2),
            )
            if maximum_delta > 0.005 or not rounded_equal:
                raise ValueError(
                    f"Refreshed lineage agreement failed parity: maximum delta {maximum_delta:.4f}."
                )
        if expected_precursor is not None:
            expected_values = expected_precursor.set_index("cell_class")[["HECTOR"] + METHODS]
            observed_values = refreshed_precursor.set_index("cell_class")[["HECTOR"] + METHODS]
            observed_values = observed_values.reindex(expected_values.index)
            maximum_delta = np.nanmax(np.abs(observed_values - expected_values).to_numpy())
            rounded_equal = np.array_equal(
                np.round(observed_values.to_numpy(), 2),
                np.round(expected_values.to_numpy(), 2),
            )
            if maximum_delta > 0.005 or not rounded_equal:
                raise ValueError(
                    f"Refreshed precursor ordering failed parity: maximum delta {maximum_delta:.4f}."
                )

        staged_cache = temporary_directory / "supplementary_cache"
        staged_cache.mkdir()
        refreshed_metadata.reset_index().to_parquet(
            staged_cache / "benchmark_cells.parquet", index=False
        )
        refreshed_orderings.reset_index().to_parquet(
            staged_cache / "benchmark_orderings.parquet", index=False
        )
        refreshed_agreement.to_csv(staged_cache / "lineage_agreement.csv", index=False)
        refreshed_precursor.to_csv(staged_cache / "precursor_order.csv", index=False)

        refreshed_sample_sha256 = hashlib.sha256(
            "\n".join(refreshed_metadata.index).encode()
        ).hexdigest()
        refreshed_manifest = {
            "schema_version": BENCHMARK_SETTINGS["schema_version"],
            "settings": BENCHMARK_SETTINGS,
            "settings_sha256": BENCHMARK_SETTINGS_SHA256,
            "main_cell_cache_sha256": main_cell_cache_sha256,
            "sample_sha256": refreshed_sample_sha256,
            "cells": int(len(refreshed_metadata)),
            "environment_executables": {
                "trajpy": str(TRAJPY_PYTHON), "traj_rscript": str(TRAJ_RSCRIPT),
            },
            "files": {},
        }
        for staged_path in sorted(staged_cache.iterdir()):
            refreshed_manifest["files"][staged_path.name] = {
                "bytes": staged_path.stat().st_size,
                "sha256": hashlib.sha256(staged_path.read_bytes()).hexdigest(),
            }
        (staged_cache / "manifest.json").write_text(
            json.dumps(refreshed_manifest, indent=2)
        )

        SUPPLEMENTARY_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for staged_path in staged_cache.iterdir():
            staged_path.replace(SUPPLEMENTARY_CACHE_DIRECTORY / staged_path.name)
        SUPPLEMENTARY_REFRESH_LOG_PATH.write_text("\n".join(refresh_log_lines))


# ## 5. Validate and load the supplementary cache
# Every cached file is fingerprinted against the main cell cache and the frozen
# benchmark settings. The figure is not drawn if any part has drifted.

required_supplementary_paths = [
    BENCHMARK_CELL_PATH, BENCHMARK_ORDERING_PATH, LINEAGE_AGREEMENT_PATH,
    PRECURSOR_ORDER_PATH, SUPPLEMENTARY_MANIFEST_PATH,
]
missing_supplementary_paths = [
    str(path) for path in required_supplementary_paths if not path.exists()
]
if missing_supplementary_paths:
    raise FileNotFoundError(
        "Missing supplementary caches. Complete the validated cache migration first: "
        f"{missing_supplementary_paths}"
    )

supplementary_manifest = json.loads(SUPPLEMENTARY_MANIFEST_PATH.read_text())
if supplementary_manifest.get("schema_version") != BENCHMARK_SETTINGS["schema_version"]:
    raise ValueError("Supplementary cache schema version does not match the script.")
if supplementary_manifest.get("settings_sha256") != BENCHMARK_SETTINGS_SHA256:
    raise ValueError("Supplementary benchmark settings fingerprint differs.")
if supplementary_manifest.get("main_cell_cache_sha256") != main_cell_cache_sha256:
    raise ValueError("Supplementary cache was built from a different HECTOR cell cache.")

supplementary_files = {
    "benchmark_cells.parquet": BENCHMARK_CELL_PATH,
    "benchmark_orderings.parquet": BENCHMARK_ORDERING_PATH,
    "lineage_agreement.csv": LINEAGE_AGREEMENT_PATH,
    "precursor_order.csv": PRECURSOR_ORDER_PATH,
}
for cache_name, cache_path in supplementary_files.items():
    observed_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    expected_sha256 = supplementary_manifest["files"][cache_name]["sha256"]
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Fingerprint mismatch for {cache_name}.")

benchmark_cells = pd.read_parquet(BENCHMARK_CELL_PATH).set_index("cell_id", drop=False)
benchmark_orderings = pd.read_parquet(BENCHMARK_ORDERING_PATH).set_index("cell_id", drop=False)
lineage_agreement = pd.read_csv(LINEAGE_AGREEMENT_PATH)
precursor_order = pd.read_csv(PRECURSOR_ORDER_PATH)

if len(benchmark_cells) != BENCHMARK_SETTINGS["cells"]:
    raise ValueError(f"Expected 30,000 benchmark cells, found {len(benchmark_cells):,}.")
if benchmark_cells.index.duplicated().any() or benchmark_orderings.index.duplicated().any():
    raise ValueError("Benchmark caches contain duplicate cell barcodes.")
if set(METHODS + ["HECTOR"]).difference(benchmark_orderings.columns):
    raise ValueError("Benchmark ordering cache does not contain all six orderings.")
if len(lineage_agreement) != 7:
    raise ValueError(f"Expected seven lineage rows, found {len(lineage_agreement)}.")
if len(precursor_order) != 5:
    raise ValueError(f"Expected five precursor/mature classes, found {len(precursor_order)}.")

benchmark_cells = benchmark_cells.join(
    benchmark_orderings.drop(columns=["cell_id"], errors="ignore"), how="inner"
)
if len(benchmark_cells) != BENCHMARK_SETTINGS["cells"]:
    raise ValueError("The benchmark metadata and ordering tables do not align exactly.")


# ## 6. Load and validate the five-edge RNA-chromatin caches
# The supplementary script draws the canonical refreshed Signac analysis. It does
# not read fragments, rerun Signac, or reinterpret MultiVelo as pseudotime.

rna_curves = pd.read_csv(RNA_CURVE_PATH)
signac_profiles = pd.read_csv(SIGNAC_PROFILE_PATH)
signac_effects = pd.read_csv(SIGNAC_EFFECT_PATH)
tabix_effects = pd.read_csv(TABIX_EFFECT_PATH)

for edge_name in EDGE_ORDER:
    for set_name in ["EARLY", "LATE"]:
        selected_curves = rna_curves[
            rna_curves["edge_key"].eq(edge_name) & rna_curves["set"].eq(set_name)
        ]
        if selected_curves["symbol"].nunique() != 100:
            raise ValueError(f"{edge_name} {set_name} does not contain 100 RNA genes.")
        selected_effects = signac_effects[
            signac_effects["edge_key"].eq(edge_name) & signac_effects["set"].eq(set_name)
        ]
        if selected_effects["donor_id"].nunique() != 6:
            raise ValueError(f"{edge_name} {set_name} does not contain six Signac donors.")

for edge_stem in ["rgc", "amacrine", "horizontal", "prpc", "nrpc"]:
    promoter_path = CACHE_DIRECTORY / f"{edge_stem}_promoters.csv"
    promoters = pd.read_csv(promoter_path)
    promoter_counts = promoters.groupby("edge_pattern")["symbol"].nunique().to_dict()
    if promoter_counts != {"CTRL": 200, "EARLY": 100, "LATE": 100}:
        raise ValueError(f"Unexpected promoter counts for {edge_stem}: {promoter_counts}")


# ## 7. Shared visual style
# All supplements use the main figure's typography, compact labels, and editable
# PDF text. PNG previews are rendered from the canonical PDFs after export.

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ## 8. Supplementary Figure 12: retinal trajectory validation
# Classification performance, five-tool rank agreement, and the complete
# precursor/mature comparison are presented as one method-validation figure.

harmonization = pd.read_csv(HARMONIZATION_PATH)
prediction_to_majorclass = dict(zip(
    harmonization["hector_label"], harmonization["majorclass"]
))
prediction_to_majorclass = {
    key: value for key, value in prediction_to_majorclass.items()
    if pd.notna(value) and str(value) != ""
}

accuracy_cells = main_cells[["majorclass", "hector_prediction"]].copy()
accuracy_cells["author_class"] = accuracy_cells["majorclass"].astype(str).replace({
    "PRPC": "RPC", "NRPC": "RPC",
})
accuracy_cells["hector_class"] = accuracy_cells["hector_prediction"].map(
    prediction_to_majorclass
)

confusion_counts = pd.crosstab(
    accuracy_cells["author_class"], accuracy_cells["hector_class"], dropna=False
).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
confusion_percent = confusion_counts.div(confusion_counts.sum(axis=1), axis=0) * 100
overall_accuracy = accuracy_cells["author_class"].eq(accuracy_cells["hector_class"]).mean()
if not np.isclose(overall_accuracy, 0.942, atol=0.001):
    raise ValueError(f"Expected approximately 94.2% accuracy, found {overall_accuracy:.3%}.")

metric_rows = []
for class_name in CLASS_ORDER:
    true_positive = (
        accuracy_cells["author_class"].eq(class_name)
        & accuracy_cells["hector_class"].eq(class_name)
    ).sum()
    false_positive = (
        ~accuracy_cells["author_class"].eq(class_name)
        & accuracy_cells["hector_class"].eq(class_name)
    ).sum()
    false_negative = (
        accuracy_cells["author_class"].eq(class_name)
        & ~accuracy_cells["hector_class"].eq(class_name)
    ).sum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)
    metric_rows.append({
        "class": class_name, "precision": precision, "recall": recall, "f1": f1_score,
    })
classification_metrics = pd.DataFrame(metric_rows)

s12_figure = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN), facecolor="white")
s12_layout = GridSpec(
    3, 12, figure=s12_figure, height_ratios=[1.0, 0.65, 1.25], hspace=0.60, wspace=0.95
)

# --- Panel a: confusion matrix.
s12_a = s12_figure.add_subplot(s12_layout[0, 0:5])
confusion_image = s12_a.imshow(
    confusion_percent.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="equal"
)
s12_a.set_xticks(range(len(CLASS_ORDER)))
s12_a.set_xticklabels(
    [CLASS_NAMES[name] for name in CLASS_ORDER], rotation=45, ha="right", fontsize=6.5
)
s12_a.set_yticks(range(len(CLASS_ORDER)))
s12_a.set_yticklabels([CLASS_NAMES[name] for name in CLASS_ORDER], fontsize=6.5)
s12_a.set_ylabel("Author annotation", fontsize=8)
for row in range(len(CLASS_ORDER)):
    for column in range(len(CLASS_ORDER)):
        value = confusion_percent.iloc[row, column]
        if confusion_counts.iloc[row, column] > 0:
            s12_a.text(
                column, row, f"{value:.0f}%", ha="center", va="center", fontsize=6,
                color="white" if value > 50 else "black",
                fontweight="bold" if row == column else "normal",
            )
s12_a.text(-0.16, 1.08, "a", transform=s12_a.transAxes, fontsize=14, fontweight="bold")
s12_colourbar = s12_figure.colorbar(
    confusion_image, ax=s12_a, fraction=0.046, pad=0.04
)
s12_colourbar.ax.set_title("%", fontsize=7, pad=3)
s12_colourbar.ax.tick_params(labelsize=6)

# --- Panel b: precision, recall, and F1.
s12_b = s12_figure.add_subplot(s12_layout[0, 7:12])
metric_x = np.arange(len(CLASS_ORDER))
metric_width = 0.25
s12_b.bar(
    metric_x - metric_width, classification_metrics["precision"], metric_width,
    label="Precision", color="#4DBBD5", edgecolor="black", linewidth=0.4,
)
s12_b.bar(
    metric_x, classification_metrics["recall"], metric_width,
    label="Recall", color="#E64B35", edgecolor="black", linewidth=0.4,
)
s12_b.bar(
    metric_x + metric_width, classification_metrics["f1"], metric_width,
    label="F1", color="#3C5488", edgecolor="black", linewidth=0.4,
)
s12_b.set_xticks(metric_x)
s12_b.set_xticklabels(
    [CLASS_NAMES[name] for name in CLASS_ORDER], rotation=45, ha="right", fontsize=6.5
)
s12_b.set_ylabel("Score")
s12_b.set_ylim(0, 1.16)
s12_b.grid(axis="y", alpha=0.25, linewidth=0.5)
s12_b.spines[["top", "right"]].set_visible(False)
s12_b.legend(
    frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=3,
)
s12_b.text(
    0.02, 0.97, f"Overall accuracy: {overall_accuracy:.1%}",
    transform=s12_b.transAxes, fontsize=7, fontweight="bold", va="top",
)
s12_b.text(-0.16, 1.08, "b", transform=s12_b.transAxes, fontsize=14, fontweight="bold")

# --- Panel c: agreement of HECTOR rank with five independent orderings.
s12_c = s12_figure.add_subplot(s12_layout[1, 1:12])
lineage_order = list(LINEAGE_NAMES)
lineage_table = lineage_agreement.set_index("lineage").reindex(lineage_order)
agreement_matrix = lineage_table[METHODS].T.to_numpy(float)
agreement_norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
agreement_image = s12_c.imshow(agreement_matrix, cmap="RdBu", norm=agreement_norm, aspect="auto")
s12_c.set_xticks(range(len(lineage_order)))
s12_c.set_xticklabels(
    [textwrap.fill(LINEAGE_NAMES[name], 13, break_long_words=False) for name in lineage_order],
    rotation=45, ha="right", fontsize=5.8,
)
s12_c.set_yticks(range(len(METHODS)))
s12_c.set_yticklabels(METHODS, fontsize=6.5)
for row in range(len(METHODS)):
    for column in range(len(lineage_order)):
        value = agreement_matrix[row, column]
        s12_c.text(
            column, row, f"{value:.2f}", ha="center", va="center", fontsize=5.5,
            color="white" if abs(value) > 0.42 else "#2A2A2A",
        )
for column, lineage_name in enumerate(lineage_order):
    s12_c.text(
        column, -0.78, f"n={int(lineage_table.loc[lineage_name, 'n']):,}",
        ha="center", va="bottom", fontsize=5.2, color="0.45",
    )
s12_c.text(-0.18, 1.08, "c", transform=s12_c.transAxes, fontsize=14, fontweight="bold")
s12_c_colourbar = s12_figure.colorbar(
    agreement_image, ax=s12_c, orientation="vertical", fraction=0.025, pad=0.025
)
s12_c_colourbar.ax.tick_params(labelsize=6)

# --- Panel d: all precursor and mature distributions, without cherry-picking.
precursor_grid = GridSpecFromSubplotSpec(
    6, 5, subplot_spec=s12_layout[2, :], hspace=0.45, wspace=0.22
)
ordering_methods = ["HECTOR"] + METHODS
histogram_edges = np.linspace(0, 1, 26)
benchmark_cells["group"] = benchmark_cells["majorclass"].astype(str).map(MAJORCLASS_TO_GROUP)
subclass_text = benchmark_cells["subclass"].astype(str)
benchmark_cells["call"] = np.where(
    subclass_text.str.contains("Precursor", case=False, na=False), "precursor",
    np.where(subclass_text.isin(["PRPC", "NRPC"]), "progenitor", "mature"),
)

for column, group_name in enumerate(PRECURSOR_GROUPS):
    group_cells = benchmark_cells[
        benchmark_cells["group"].eq(group_name) & ~benchmark_cells["call"].eq("progenitor")
    ]
    for row, method_name in enumerate(ordering_methods):
        distribution_axis = s12_figure.add_subplot(precursor_grid[row, column])
        distribution_data = group_cells[[method_name, "call"]].dropna()
        percentile_rank = distribution_data[method_name].rank(pct=True)
        for call_name, call_colour in PRECURSOR_COLOURS.items():
            call_values = percentile_rank[distribution_data["call"].eq(call_name)]
            distribution_axis.hist(
                call_values, bins=histogram_edges, density=True,
                color=call_colour, alpha=0.55, linewidth=0,
            )
        precursor_values = percentile_rank[distribution_data["call"].eq("precursor")]
        mature_values = percentile_rank[distribution_data["call"].eq("mature")]
        correctly_ordered = 1.0 - mannwhitneyu(
            precursor_values, mature_values
        ).statistic / (len(precursor_values) * len(mature_values))
        distribution_axis.text(
            0.5, 1.01, f"{correctly_ordered:.2f}", transform=distribution_axis.transAxes,
            ha="center", va="bottom", fontsize=5.5, fontweight="bold",
            color="#1F1F1F" if correctly_ordered >= 0.5 else "#B3282B",
        )
        distribution_axis.set_xlim(0, 1)
        distribution_axis.set_yticks([])
        distribution_axis.spines[["top", "right", "left"]].set_visible(False)
        if row == len(ordering_methods) - 1:
            distribution_axis.set_xticks([0, 0.5, 1])
            distribution_axis.set_xticklabels(
                ["first", "", "last"] if column == 0 else [], fontsize=5
            )
        else:
            distribution_axis.set_xticks([])
        if column == 0:
            distribution_axis.set_ylabel(
                method_name, rotation=0, ha="right", va="center", fontsize=6, labelpad=4
            )
        if row == 0:
            distribution_axis.text(
                0.5, 1.55, textwrap.fill(group_name, 18),
                transform=distribution_axis.transAxes, ha="center", va="bottom", fontsize=6.5,
            )
            if column == 0:
                distribution_axis.text(
                    -0.24, 2.05, "d", transform=distribution_axis.transAxes,
                    fontsize=14, fontweight="bold", va="top",
                )
        if row == len(ordering_methods) - 1 and column == 2:
            distribution_axis.text(
                0.5, -0.55, "Position in the ordering", transform=distribution_axis.transAxes,
                ha="center", va="top", fontsize=8,
            )

s12_legend = [
    mpl.patches.Patch(facecolor=PRECURSOR_COLOURS["precursor"], alpha=0.55, label="precursor"),
    mpl.patches.Patch(facecolor=PRECURSOR_COLOURS["mature"], alpha=0.55, label="mature"),
]
s12_figure.legend(
    handles=s12_legend, loc="lower center", bbox_to_anchor=(0.5, 0.165),
    ncol=2, frameon=False, fontsize=7,
)
# Tighten the occupied region while retaining the full page and printed type sizes.
s12_figure.subplots_adjust(top=0.955, bottom=0.22, left=0.17, right=0.955)
standardize_figure(s12_figure)
s12_temporary_pdf = S12_PDF_PATH.with_suffix(".staging.pdf")
s12_figure.savefig(s12_temporary_pdf, bbox_inches=None, dpi=EXPORT_DPI)
plt.close(s12_figure)
s12_temporary_pdf.replace(S12_PDF_PATH)

s12_document = fitz.open(S12_PDF_PATH)
if len(s12_document) != 1:
    raise ValueError("Supplementary Figure 12 must contain one page.")
s12_pixmap = s12_document[0].get_pixmap(dpi=EXPORT_DPI, alpha=False)
s12_temporary_png = S12_PNG_PATH.with_suffix(".staging.png")
s12_pixmap.save(s12_temporary_png)
s12_document.close()
s12_temporary_png.replace(S12_PNG_PATH)


# ## 9. Supplementary Figure 13: subtype resolution
# Each column compares a biologically related subtype pair across the same ten
# developmental weeks and the same verified HECTOR edge coordinates.
#
# Both members of a pair must sit on edges that leave the same immediate parent
# term, because relative_position is rescaled to [0, 1] within each edge and is
# not comparable between edges of different depth. This excludes midget vs. ON
# parasol and S cone vs. retinal cone cell, whose parents sit at different
# depths; the ontology offers no depth-matched replacement pair with enough
# cells for either.

SUBTYPE_GROUPS = [
    {
        "label": "horizontal cells",
        "subtypes": [("H1 horizontal cell", "H1"), ("H2 horizontal cell", "H2")],
        "colours": ["#6B4F92", "#C6B4E4"],
    },
    {
        "label": "amacrine cells",
        "subtypes": [("GABAergic amacrine cell", "GABAergic"), ("glycinergic amacrine cell", "Glycinergic")],
        "colours": ["#38618F", "#A8C5E6"],
    },
]

subtype_cells = main_cells[
    main_cells["trajectory_state"].isin(["Stable", "Transitioning"])
    & main_cells["relative_position"].notna()
].copy()
subtype_cells["week"] = subtype_cells["development_stage"].astype(str).str.extract(
    r"(\d+)(?:st|nd|rd|th)\s+week"
)[0]
subtype_cells = subtype_cells[subtype_cells["week"].notna()].copy()
subtype_cells["week"] = subtype_cells["week"].astype(int)
development_weeks = sorted(subtype_cells["week"].unique())
if development_weeks != [9, 11, 12, 13, 14, 15, 17, 20, 21, 24]:
    raise ValueError(f"Unexpected developmental weeks: {development_weeks}")

s13_figure = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN), facecolor="white")
s13_layout = GridSpec(
    1, 2 * len(SUBTYPE_GROUPS), figure=s13_figure,
    width_ratios=[1.0, 0.22] * len(SUBTYPE_GROUPS), wspace=0.08,
)
s13_x_grid = np.linspace(0, 1, 300)

for group_index, subtype_group in enumerate(SUBTYPE_GROUPS):
    subtype_axis = s13_figure.add_subplot(s13_layout[0, group_index * 2])
    count_axis = s13_figure.add_subplot(s13_layout[0, group_index * 2 + 1])
    for week_index, week in enumerate(development_weeks):
        ridge_y = len(development_weeks) - 1 - week_index
        week_cells = subtype_cells[subtype_cells["week"].eq(week)]
        week_counts = []
        for subtype_index, (target_name, short_name) in enumerate(subtype_group["subtypes"]):
            target_cells = week_cells[week_cells["trajectory_target"].eq(target_name)]
            week_counts.append(len(target_cells))
            donor_resamples = []
            subtype_rng = np.random.default_rng(week * 100 + subtype_index)
            for donor_name, donor_cells in target_cells.groupby("donor_id", observed=True):
                donor_positions = donor_cells["relative_position"].dropna().to_numpy()
                if len(donor_positions):
                    donor_resamples.append(
                        subtype_rng.choice(donor_positions, size=500, replace=True)
                    )
            resampled_positions = np.concatenate(donor_resamples) if donor_resamples else np.array([])
            if len(resampled_positions) >= 5 and np.ptp(resampled_positions) > 0:
                subtype_density = gaussian_kde(resampled_positions)(s13_x_grid)
                subtype_density = subtype_density / subtype_density.max() * 0.45
            else:
                subtype_density = np.zeros_like(s13_x_grid)
            subtype_colour = subtype_group["colours"][subtype_index]
            subtype_axis.fill_between(
                s13_x_grid, ridge_y, ridge_y + subtype_density,
                color=subtype_colour, alpha=0.22, linewidth=0,
            )
            subtype_axis.plot(
                s13_x_grid, ridge_y + subtype_density,
                color=subtype_colour, alpha=0.85, linewidth=0.75,
            )
        count_text = " / ".join(
            f"{count / 1000:.1f}k" if count >= 1000 else str(count)
            for count in week_counts
        )
        count_axis.text(0.02, ridge_y, count_text, fontsize=5.2, color="0.35", va="center")

    subtype_axis.set_title(subtype_group["label"], fontsize=10, fontweight="bold", pad=18)
    subtype_axis.set_xlim(0, 1)
    subtype_axis.set_ylim(-0.15, len(development_weeks) - 0.35)
    subtype_axis.set_xticks([0, 0.5, 1])
    subtype_axis.set_yticks(range(len(development_weeks)))
    subtype_axis.set_yticklabels(
        [f"W{week}" for week in reversed(development_weeks)] if group_index == 0 else []
    )
    subtype_axis.tick_params(labelsize=8, length=2)
    subtype_axis.grid(axis="x", color="0.91", linewidth=0.5)
    subtype_axis.spines[["top", "right"]].set_visible(False)
    if group_index == 0:
        subtype_axis.set_ylabel("Post-conception week", fontsize=9)
    subtype_handles = [
        mpl.patches.Patch(
            facecolor=subtype_group["colours"][subtype_index], alpha=0.5,
            label=subtype_group["subtypes"][subtype_index][1],
        )
        for subtype_index in range(2)
    ]
    subtype_axis.legend(
        handles=subtype_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, fontsize=7, frameon=False, borderaxespad=0,
        handlelength=1.0, handletextpad=0.4, columnspacing=1.4,
    )
    count_axis.set_xlim(0, 1)
    count_axis.set_ylim(subtype_axis.get_ylim())
    count_axis.set_xticks([])
    count_axis.set_yticks([])
    count_axis.set_title("n", fontsize=7, color="0.35", pad=5)
    for count_spine in count_axis.spines.values():
        count_spine.set_visible(False)

s13_figure.text(0.5, 0.515, "Trajectory position", ha="center", fontsize=9)
# Occupy only the top half without changing the page or final-print text sizes.
s13_figure.subplots_adjust(
    top=0.93, bottom=0.55, left=0.12, right=0.96,
)
standardize_figure(s13_figure)
s13_temporary_pdf = S13_PDF_PATH.with_suffix(".staging.pdf")
s13_figure.savefig(s13_temporary_pdf, bbox_inches=None, dpi=EXPORT_DPI)
plt.close(s13_figure)
s13_temporary_pdf.replace(S13_PDF_PATH)

s13_document = fitz.open(S13_PDF_PATH)
if len(s13_document) != 1:
    raise ValueError("Supplementary Figure 13 must contain one page.")
s13_pixmap = s13_document[0].get_pixmap(dpi=EXPORT_DPI, alpha=False)
s13_temporary_png = S13_PNG_PATH.with_suffix(".staging.png")
s13_pixmap.save(s13_temporary_png)
s13_document.close()
s13_temporary_png.replace(S13_PNG_PATH)


# ## 10. Supplementary Figure 14: RNA-chromatin generalization
# The four non-amacrine edges use the same program definition and control-
# adjusted Signac display as the amacrine result already shown in main Figure 5.

signac_profiles["bin"] = (
    signac_profiles["offset_from_tss"] // 50
) * 50
rebinned_profiles = signac_profiles.groupby(
    ["edge_key", "donor_id", "set", "group", "bin"], as_index=False
)["mean_cpm_per_gene"].sum()
donor_averaged_profiles = rebinned_profiles.groupby(
    ["edge_key", "set", "group", "bin"], as_index=False
)["mean_cpm_per_gene"].mean()
control_profiles = donor_averaged_profiles[donor_averaged_profiles["set"].eq("CTRL")][
    ["edge_key", "group", "bin", "mean_cpm_per_gene"]
].rename(columns={"mean_cpm_per_gene": "control_cpm"})
adjusted_profiles = donor_averaged_profiles.merge(
    control_profiles, on=["edge_key", "group", "bin"], how="left"
)
adjusted_profiles["target_minus_control"] = (
    adjusted_profiles["mean_cpm_per_gene"] - adjusted_profiles["control_cpm"]
)
quarter_means = adjusted_profiles.groupby(
    ["edge_key", "set", "bin"], as_index=False
)["target_minus_control"].mean().rename(
    columns={"target_minus_control": "quarter_mean"}
)
adjusted_profiles = adjusted_profiles.merge(
    quarter_means, on=["edge_key", "set", "bin"], how="left"
)
adjusted_profiles["centered_height"] = (
    adjusted_profiles["target_minus_control"] - adjusted_profiles["quarter_mean"]
)

s14_figure = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN), facecolor="white")
s14_layout = GridSpec(
    5, 3, figure=s14_figure,
    height_ratios=[1, 1, 1, 1, 1.10], hspace=0.80, wspace=0.70,
)

for edge_index, edge_name in enumerate(S14_EDGE_ORDER):
    rna_axis = s14_figure.add_subplot(s14_layout[edge_index, 0])
    edge_rna = rna_curves[rna_curves["edge_key"].eq(edge_name)]
    for set_name in ["EARLY", "LATE"]:
        program_colourmap = PROGRAM_COLOURMAPS[set_name]
        program_curves = edge_rna[edge_rna["set"].eq(set_name)]
        for gene_name, gene_curve in program_curves.groupby("symbol"):
            gene_curve = gene_curve.sort_values("bin")
            rna_axis.plot(
                gene_curve["edge_position"], gene_curve["scaled_expression"],
                color=program_colourmap(0.55), alpha=0.12, linewidth=0.3,
            )
        mean_curve = program_curves.groupby("bin", as_index=False).agg(
            edge_position=("edge_position", "first"),
            scaled_expression=("scaled_expression", "mean"),
        ).sort_values("bin")
        mean_points = mean_curve[["edge_position", "scaled_expression"]].to_numpy()
        mean_segments = np.concatenate(
            [mean_points[:-1, None, :], mean_points[1:, None, :]], axis=1
        )
        mean_gradient = LineCollection(
            mean_segments, cmap=program_colourmap,
            norm=mcolors.Normalize(0, 1), linewidth=1.2,
        )
        mean_gradient.set_array(mean_curve["edge_position"].to_numpy())
        rna_axis.add_collection(mean_gradient)
        label_y_position = 1.02 if set_name == "LATE" else 0.03
        rna_axis.text(
            0.96,
            label_y_position,
            "turning on" if set_name == "LATE" else "turning off",
            transform=rna_axis.transAxes,
            fontsize=7,
            color=program_colourmap(0.75),
            va="bottom",
            ha="right",
        )
    rna_axis.set_xlim(0, 1)
    rna_axis.set_ylim(-0.05, 1.05)
    rna_axis.spines[["top", "right"]].set_visible(False)
    rna_axis.set_ylabel(f"{EDGE_LABELS[edge_name]}\nScaled RNA expression", fontsize=7)
    if edge_index == 0:
        rna_axis.set_title("RNA programs", fontsize=9, fontweight="semibold", pad=14)
        rna_axis.text(-0.20, 1.08, "a", transform=rna_axis.transAxes, fontsize=14, fontweight="bold")
    rna_axis.set_xlabel("Position along the edge (rank)")

    for column_index, set_name in enumerate(["EARLY", "LATE"], start=1):
        chromatin_axis = s14_figure.add_subplot(s14_layout[edge_index, column_index])
        program_colourmap = PROGRAM_COLOURMAPS[set_name]
        edge_profiles = adjusted_profiles[
            adjusted_profiles["edge_key"].eq(edge_name)
            & adjusted_profiles["set"].eq(set_name)
        ]
        for group_name in ["Q1", "Q2", "Q3", "Q4"]:
            group_curve = edge_profiles[edge_profiles["group"].eq(group_name)].sort_values("bin")
            chromatin_axis.plot(
                group_curve["bin"] / 1_000,
                gaussian_filter1d(group_curve["centered_height"].to_numpy(), sigma=1.0),
                color=program_colourmap(QUARTER_COLOUR_POSITION[set_name][group_name]),
                linewidth=0.75, label=group_name,
            )
        chromatin_axis.axvline(0, color="0.6", linestyle="--", linewidth=0.65)
        chromatin_axis.axhline(0, color="0.85", linewidth=0.55)
        data_y_min, data_y_max = chromatin_axis.get_ylim()
        chromatin_axis.set_ylim(
            data_y_min,
            data_y_max + 0.30 * (data_y_max - data_y_min),
        )
        chromatin_axis.spines[["top", "right"]].set_visible(False)
        if edge_index == 0:
            chromatin_axis.set_title(
                "Chromatin\ngenes turning off" if set_name == "EARLY"
                else "Chromatin\ngenes turning on",
                fontsize=9, fontweight="semibold",
            )
        if column_index == 1:
            chromatin_axis.set_ylabel("Tn5 CPM, minus control\nand quarter mean", fontsize=7)
        chromatin_axis.set_xlabel("Distance from TSS (kb)")
        if edge_index == 0:
            chromatin_axis.legend(
                title="edge-position quarter", fontsize=6, title_fontsize=6,
                loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=2,
                handlelength=1.0, frameon=False, borderaxespad=0,
            )

# --- Panel b: donor-level Signac effects and independent tabix means.
effect_layout = GridSpecFromSubplotSpec(
    1, 2, subplot_spec=s14_layout[4, :], wspace=0.65
)
edge_y = np.arange(len(S14_EDGE_ORDER))[::-1]
for summary_index, set_name in enumerate(["EARLY", "LATE"]):
    effect_axis = s14_figure.add_subplot(effect_layout[0, summary_index])
    for edge_index, edge_name in enumerate(S14_EDGE_ORDER):
        y_value = edge_y[edge_index]
        donor_effects = signac_effects[
            signac_effects["edge_key"].eq(edge_name)
            & signac_effects["set"].eq(set_name)
        ]["target_minus_control"].to_numpy()
        tabix_values = tabix_effects[
            tabix_effects["edge_key"].eq(edge_name)
            & tabix_effects["set"].eq(set_name)
        ]["target_minus_control"].to_numpy()
        donor_jitter = np.linspace(-0.12, 0.12, len(donor_effects))
        point_colour = PROGRAM_COLOURMAPS[set_name](0.60)
        effect_axis.scatter(
            donor_effects, y_value + donor_jitter, s=7,
            color=point_colour, alpha=0.55, linewidth=0,
        )
        signac_mean = donor_effects.mean()
        signac_se = donor_effects.std(ddof=1) / np.sqrt(len(donor_effects))
        effect_axis.errorbar(
            signac_mean, y_value, xerr=signac_se,
            fmt="o", color="black", markersize=2.8, capsize=1.5, linewidth=0.6,
        )
        effect_axis.scatter(
            tabix_values.mean(), y_value, marker="D", s=14,
            facecolor="white", edgecolor=point_colour, linewidth=0.75, zorder=4,
        )
    effect_axis.axvline(0, color="0.65", linewidth=0.7, linestyle="--")
    effect_axis.set_yticks(edge_y)
    effect_axis.set_yticklabels([EDGE_LABELS[edge] for edge in S14_EDGE_ORDER], fontsize=7)
    effect_axis.set_xlabel("Target minus matched-control effect")
    effect_axis.set_title(
        "Genes turning off" if set_name == "EARLY" else "Genes turning on",
        fontsize=9, fontweight="semibold",
    )
    effect_axis.spines[["top", "right"]].set_visible(False)
    if summary_index == 0:
        effect_axis.text(-0.15, 1.10, "b", transform=effect_axis.transAxes, fontsize=14, fontweight="bold")

effect_legend = [
    Line2D([], [], marker="o", linestyle="None", color="0.35", markersize=2.6, label="Signac donor"),
    Line2D([], [], marker="o", linestyle="None", color="black", markersize=2.8, label="Signac mean +/- SE"),
    Line2D([], [], marker="D", linestyle="None", markerfacecolor="white", markeredgecolor="0.35", markeredgewidth=0.75, markersize=3.7, label="tabix mean"),
]
s14_figure.legend(
    handles=effect_legend, loc="lower center", bbox_to_anchor=(0.5, 0.012),
    ncol=3, frameon=False, fontsize=7,
)
s14_figure.subplots_adjust(top=0.94, bottom=0.09, left=0.14, right=0.97)
standardize_figure(s14_figure)
s14_temporary_pdf = S14_PDF_PATH.with_suffix(".staging.pdf")
s14_figure.savefig(s14_temporary_pdf, bbox_inches=None, dpi=EXPORT_DPI)
plt.close(s14_figure)
s14_temporary_pdf.replace(S14_PDF_PATH)

s14_document = fitz.open(S14_PDF_PATH)
if len(s14_document) != 1:
    raise ValueError("Supplementary Figure 14 must contain one page.")
s14_pixmap = s14_document[0].get_pixmap(dpi=EXPORT_DPI, alpha=False)
s14_temporary_png = S14_PNG_PATH.with_suffix(".staging.png")
s14_pixmap.save(s14_temporary_png)
s14_document.close()
s14_temporary_png.replace(S14_PNG_PATH)


# ## 11. Numerical and artifact parity report
# Record the quantities represented in the figures and verify that the canonical
# PDFs are readable, single-page documents with nonempty text and drawing bounds.

pdf_paths = [S12_PDF_PATH, S13_PDF_PATH, S14_PDF_PATH]
pdf_checks = []
for pdf_path in pdf_paths:
    pdf_document = fitz.open(pdf_path)
    pdf_text = "".join(page.get_text() for page in pdf_document)
    pdf_checks.append({
        "path": str(pdf_path.relative_to(FIGURE5_DIRECTORY)),
        "pages": len(pdf_document),
        "bytes": pdf_path.stat().st_size,
        "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "text_characters": len(pdf_text),
    })
    if len(pdf_document) != 1 or not pdf_text.strip():
        raise ValueError(f"Invalid final PDF: {pdf_path}")
    pdf_document.close()

parity_report = {
    "schema_version": 1,
    "main_cache_created": main_manifest.get("created"),
    "main_cell_cache_sha256": main_cell_cache_sha256,
    "supplementary_settings_sha256": BENCHMARK_SETTINGS_SHA256,
    "atlas_cells": int(len(main_cells)),
    "benchmark_cells": int(len(benchmark_cells)),
    "benchmark_sample_sha256": supplementary_manifest["sample_sha256"],
    "overall_accuracy": float(overall_accuracy),
    "lineage_rows": int(len(lineage_agreement)),
    "precursor_classes": int(len(precursor_order)),
    "rna_gene_counts": [
        {"edge_key": edge_key, "set": set_name, "genes": int(gene_count)}
        for (edge_key, set_name), gene_count in
        rna_curves.groupby(["edge_key", "set"])["symbol"].nunique().items()
    ],
    "signac_donor_counts": [
        {"edge_key": edge_key, "set": set_name, "donors": int(donor_count)}
        for (edge_key, set_name), donor_count in
        signac_effects.groupby(["edge_key", "set"])["donor_id"].nunique().items()
    ],
    "pdfs": pdf_checks,
    "python": sys.version,
    "packages": {
        package_name: importlib.metadata.version(package_name)
        for package_name in ["pandas", "numpy", "matplotlib", "scipy", "PyMuPDF"]
    },
}
SUPPLEMENTARY_PARITY_PATH.write_text(json.dumps(parity_report, indent=2, default=str))

print(f"Wrote {S12_PDF_PATH}")
print(f"Wrote {S13_PDF_PATH}")
print(f"Wrote {S14_PDF_PATH}")
print(f"Parity report: {SUPPLEMENTARY_PARITY_PATH}")
