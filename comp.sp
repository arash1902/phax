***Threshold-Detection Comparator

.subckt inv in out vdd
.lib 'crn90g_2d5_lk_v1d2p1.l' tt_lvt
.param lmin=100nm
m1 out in 0 	0		nch_hvt	 w='1*Lmin' L='Lmin'
m2 out in vdd	vdd		pch_hvt	 w='2*Lmin' L='Lmin'
.ends 
.subckt inv_diff in out vdd vss 
.lib 'crn90g_2d5_lk_v1d2p1.l' tt_lvt
.param sz = 1
.param lmin=100nm
m1 out in vss 	vss		nch_lvt	 w='1.2*Lmin*sz' L='Lmin'
m2 out in vdd	vdd		pch_lvt	 w='2*Lmin*sz' L='Lmin'
.ends 


*.subckt inv_diff_finfet in out vdd vss 
*.lib 'finfet/finfet.lib' ptm20lstp
*.param sz = 1
*.param lmin='lg'
*xm1 out in vss 	vss		nfet_lstp	 nfin='sz' L='Lmin'
*xm2 out in vdd	vdd		pfet_lstp	 nfin='sz' L='Lmin'
*.ends 
**************Stage1************************
.subckt comp vx vy x2 vdd
.param lmin=100nm
.param wnb=2
+wpb=3
+wns1=8
+wns2=3
+wps11=1.2
+wps12=7.2
+wps21=1.2
+wps22=8.2
+wns10=6
+wns20=9
+wnI=3
+wpI=9
ms11 x1 Vx x10 0 nch_hvt w='wns1*Lmin' L='2*Lmin'
ms12 x2 Vy x10 0 nch_hvt w='wns1*Lmin' L='2*Lmin'
ms13 x1 x1 vdd vdd pch_hvt w='wps11*Lmin' L='3.4*Lmin'
ms14 x2 x2 vdd vdd pch_hvt w='wps11*Lmin' L='3.4*Lmin'
ms15 x1 vpbias vdd vdd pch_hvt w='wps12*Lmin' L='2.3*Lmin' m=1
ms16 x2 vpbias vdd vdd pch_hvt w='wps12*Lmin' L='2.3*Lmin' m=1
ms10 x10 vnbias 0 0 nch_hvt w='wns10*Lmin' L='Lmin' m=2
Ibias vdd vnbias 4u
mb1 vnbias vnbias 0 0 nch_hvt w='wnb*Lmin' L='Lmin'
mb2 vpbias vnbias 0 0 nch_hvt w='wnb*Lmin' L='Lmin'
mb3 vpbias vpbias vdd vdd pch_hvt w='wpb*Lmin' L='Lmin' m=2

.ends