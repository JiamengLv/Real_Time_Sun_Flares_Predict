# Real_Time_Solar_Flares_Predict

https://nadc.china-vo.org/flarecast/

多波段（hmi,aia）太阳活动区的标准数据集制作：
     
      1.使用交互式页面得到活动区信息，参考：https://github.com/DuckDuckPig/AR-flare
      2.FITS_CEA_AIA_HMI: 使用交互式页面得到带有活动区信息的数据之后，使用sunpy.map 对 HMI的数据和aia的数据 进行投影转换（CEA）和裁剪，实现论文中的效果。
