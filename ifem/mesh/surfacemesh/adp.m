function z = adp(p)

A=[4.833   6.136   3.456    1.500
 3.831   7.111   3.945    1.850
 4.383   8.179   4.757    1.500
 2.852   7.638   2.998    1.500
 2.993   6.301   4.973    1.500
 2.455   4.962   4.617    1.850
 2.052   4.692   3.208    1.500
 3.095   3.851   5.402    1.500
 1.029   5.115   5.248    1.500
 0.793   5.334   6.641    1.700
 0.386   6.333   6.797    1.200
 1.671   5.152   7.261    1.200
-0.244   4.403   7.142    1.700
-0.099   4.489   8.219    1.200
 0.063   2.920   6.799    1.700
 0.338   2.810   5.750    1.200
 0.947   2.272   7.556    1.500
 0.635   1.374   7.686    1.200
-1.368   2.309   6.996    1.700
-1.634   1.624   6.191    1.200
-1.474   1.667   8.255    1.500
-1.087   2.252   8.910    1.200
-1.600   4.698   6.853    1.500
-2.435   3.502   7.052    1.700
-2.929   3.453   8.022    1.200
-3.511   3.387   5.982    1.550
-3.468   3.225   4.628    1.700
-2.539   3.245   4.078    1.200
-4.587   3.150   4.028    1.550
-5.492   3.149   5.051    1.700
-4.824   3.217   6.255    1.700
-5.443   3.243   7.493    1.550
-6.747   3.148   7.400    1.700
-7.269   3.168   8.345    1.200
-7.483   3.049   6.334    1.550
-6.889   2.991   5.177    1.700
-7.756   2.894   4.149    1.550
-8.750   2.827   4.316    1.200
-7.301   2.850   3.248    1.200];

c = A(:,1:3);
r = A(:, 4);
