import torch

heat_eq_inputs = [
    torch.tensor(
        [
            198.8556,
            846.0942,
            855.0133,
            753.9380,
            805.3965,
            413.7646,
            469.4816,
            585.4802,
            416.3366,
            689.3568,
            553.0941,
            514.5001,
            666.0671,
            972.3352,
            619.9268,
            477.4501,
            891.8897,
            639.9215,
            205.6411,
            55.0707,
            861.3621,
            339.2696,
            547.4985,
            466.1339,
            832.0196,
            522.4506,
            260.4631,
            538.0689,
            603.9344,
            734.3096,
            397.1854,
            394.2798,
            552.0857,
            139.6251,
            65.9723,
            449.0759,
            493.5453,
            345.2740,
            180.8261,
            891.9362,
            847.7637,
            543.9079,
            71.1120,
            33.2777,
            714.6210,
            770.9079,
            330.3165,
            560.5562,
            948.7731,
            565.3021,
            202.5863,
            870.0583,
            29.4966,
            791.8011,
            509.1472,
            398.3628,
            946.8214,
            736.1155,
            236.8429,
            140.9616,
            139.9883,
            987.2784,
            114.3440,
            617.3469,
            186.4265,
            132.6854,
            238.5827,
            643.6541,
            193.5946,
            511.8262,
            27.2019,
            845.0001,
            182.0614,
            450.6221,
            860.8644,
            252.0624,
            246.5792,
            26.5824,
            523.1374,
            140.1638,
            719.5886,
            394.8317,
            343.4214,
            451.7281,
            607.3315,
            785.2297,
            785.0710,
            520.9001,
            535.0784,
            535.6682,
            162.6431,
            641.8715,
            568.8826,
            541.0428,
            583.8602,
            298.4215,
            395.0290,
            47.6823,
            150.0522,
            322.3262,
            100.2375,
        ]
    ),
    torch.tensor(
        [
            246.3110,
            346.9463,
            920.6794,
            699.6705,
            557.1841,
            212.1623,
            610.0323,
            767.9116,
            644.8337,
            89.8654,
            99.1227,
            350.6152,
            652.8633,
            370.5943,
            228.7001,
            703.2086,
            315.2505,
            868.2393,
            336.0684,
            4.9843,
            928.6802,
            783.7191,
            84.7018,
            78.2264,
            551.9172,
            426.8420,
            525.3440,
            865.8888,
            885.4938,
            666.6174,
            380.7350,
            591.0228,
            668.2376,
            105.1518,
            420.0271,
            312.8023,
            963.8856,
            537.2138,
            77.9587,
            769.1694,
            474.6398,
            451.1705,
            659.3651,
            595.3548,
            123.9251,
            603.0153,
            438.1218,
            316.1721,
            489.6260,
            677.6182,
            128.3541,
            529.0106,
            5.4171,
            813.8585,
            416.4025,
            386.8340,
            241.7890,
            860.6790,
            626.7441,
            178.4477,
            622.1263,
            231.7988,
            721.3308,
            711.6857,
            569.7891,
            403.9121,
            397.2841,
            704.7053,
            77.0218,
            550.7336,
            196.0004,
            627.7814,
            204.2814,
            407.7465,
            288.6496,
            565.1867,
            460.8380,
            881.9117,
            294.2352,
            219.2393,
            922.2241,
            647.9928,
            808.1757,
            315.7180,
            187.4690,
            429.7380,
            591.3821,
            210.2065,
            350.7801,
            749.0131,
            116.3160,
            216.7428,
            179.8562,
            403.8709,
            155.3422,
            135.4688,
            184.6709,
            633.5441,
            12.7196,
            309.9612,
            808.8348,
        ]
    ),
    torch.tensor(
        [
            843.8328,
            372.2674,
            794.9582,
            691.7625,
            336.0793,
            118.8168,
            769.2312,
            380.4677,
            364.7382,
            224.4297,
            218.7282,
            149.8966,
            79.7839,
            138.5943,
            161.6942,
            364.4291,
            397.1774,
            211.3882,
            216.8414,
            157.2145,
            589.5347,
            120.2346,
            366.0268,
            825.2930,
            328.5240,
            839.5279,
            611.9987,
            268.4572,
            472.5786,
            496.0073,
            797.3116,
            710.4428,
            786.9168,
            155.4209,
            258.8429,
            207.6805,
            432.7140,
            176.2611,
            147.8430,
            686.3946,
            225.8061,
            541.1881,
            631.6400,
            499.4539,
            928.5876,
            34.6580,
            825.4531,
            899.1171,
            681.1959,
            38.4575,
            544.0702,
            280.2285,
            169.8590,
            640.9733,
            633.0620,
            949.1155,
            615.8621,
            736.5054,
            389.0420,
            220.6755,
            312.3369,
            223.3599,
            632.7808,
            713.7342,
            285.4288,
            727.9604,
            137.5046,
            503.2483,
            720.0152,
            675.5729,
            614.0613,
            78.7715,
            880.7277,
            124.8147,
            950.3826,
            776.6353,
            826.3371,
            351.5152,
            873.6580,
            516.0698,
            376.4881,
            744.5516,
            431.2473,
            554.6046,
            729.0397,
            29.8517,
            298.7455,
            667.3636,
            892.4149,
            584.4640,
            578.2502,
            261.0150,
            849.1693,
            833.9897,
            548.9365,
            254.4293,
            857.1882,
            116.2218,
            480.8369,
            848.4462,
            215.3205,
        ]
    ),
    torch.tensor(
        [
            955.3790,
            301.4926,
            789.3935,
            614.0554,
            883.1428,
            166.1508,
            670.0114,
            797.3592,
            838.6680,
            362.3044,
            606.6512,
            268.0467,
            943.7609,
            103.7034,
            826.9409,
            188.7289,
            236.0271,
            265.3865,
            594.4973,
            59.0782,
            303.1123,
            198.6063,
            714.9880,
            282.5895,
            752.2621,
            425.3362,
            911.2921,
            270.2863,
            559.6039,
            210.5311,
            596.9486,
            748.4596,
            323.8044,
            898.6643,
            503.9744,
            698.2272,
            430.4587,
            392.4307,
            50.8310,
            529.4579,
            260.1289,
            290.4930,
            111.4179,
            456.3513,
            916.6769,
            450.2050,
            743.7144,
            589.2620,
            921.7917,
            592.9883,
            593.8981,
            328.7985,
            441.7155,
            841.4966,
            971.9990,
            506.8242,
            381.4337,
            212.9990,
            476.7168,
            572.6101,
            812.2165,
            664.9412,
            75.5513,
            722.7299,
            211.2869,
            605.2353,
            971.8937,
            824.3165,
            210.4465,
            108.0469,
            909.3866,
            793.8533,
            919.9367,
            325.6106,
            408.9444,
            457.7509,
            762.3027,
            434.4428,
            171.1811,
            831.6678,
            659.7897,
            920.1708,
            490.4767,
            59.8509,
            611.1643,
            208.1883,
            45.2970,
            341.7631,
            744.3889,
            264.6651,
            183.6062,
            413.5320,
            297.4626,
            269.4822,
            93.1290,
            901.1049,
            198.9354,
            787.2162,
            63.0373,
            613.8838,
            487.8021,
        ]
    ),
    torch.tensor(
        [
            576.5198,
            395.5631,
            634.1282,
            920.1050,
            605.0273,
            581.7570,
            739.7795,
            655.3798,
            571.5762,
            488.0795,
            825.1696,
            180.6603,
            913.6379,
            230.5515,
            462.3380,
            367.0402,
            443.8139,
            201.4107,
            508.0470,
            101.4277,
            588.5871,
            552.6096,
            873.6124,
            10.4362,
            95.8297,
            126.0317,
            773.0066,
            831.1926,
            581.1642,
            703.5735,
            20.3960,
            968.4156,
            312.8857,
            501.9214,
            13.0342,
            736.0634,
            128.6287,
            401.3460,
            686.3420,
            704.4687,
            468.6576,
            892.0625,
            708.9551,
            402.7410,
            181.1224,
            780.3846,
            323.6324,
            975.3024,
            888.3619,
            249.4463,
            323.8891,
            794.2135,
            653.9296,
            642.4874,
            594.0300,
            83.2704,
            676.9639,
            433.5136,
            58.7468,
            598.4786,
            141.4465,
            651.9641,
            939.8253,
            672.3701,
            476.5230,
            599.6072,
            599.5643,
            899.1403,
            785.9212,
            680.9412,
            798.9107,
            519.0413,
            531.4354,
            707.9305,
            139.5806,
            872.4008,
            496.2944,
            883.9675,
            590.6477,
            745.5620,
            928.3616,
            943.7937,
            346.3990,
            594.6929,
            579.7094,
            996.9269,
            914.7407,
            274.4537,
            381.5956,
            734.9935,
            212.2648,
            344.7945,
            849.4138,
            168.6927,
            465.5395,
            956.4512,
            971.9449,
            746.8204,
            47.7553,
            516.1475,
            608.5712,
        ]
    ),
]

heat_eq_outputs = [
    torch.tensor(
        [
            67.6956,
            313.4373,
            394.4938,
            399.1351,
            352.4915,
            257.3163,
            266.3477,
            304.1212,
            341.3968,
            371.3360,
            338.1879,
            404.0784,
            642.8170,
            839.1017,
            835.4871,
            838.5233,
            808.5266,
            730.9728,
            681.6769,
            748.6910,
            840.9599,
            815.3228,
            728.4275,
            647.3047,
            617.2127,
            605.4141,
            571.7966,
            683.2874,
            732.2680,
            632.4385,
            506.4599,
            539.4362,
            517.8021,
            329.3401,
            180.1618,
            169.9467,
            195.8306,
            216.3683,
            314.5316,
            448.5602,
            463.1028,
            318.9000,
            217.7137,
            279.0336,
            444.5768,
            463.7382,
            405.2247,
            468.7990,
            517.5533,
            484.6900,
            483.1098,
            569.3979,
            588.3190,
            624.9644,
            678.3881,
            696.8560,
            665.0057,
            394.9656,
            85.1657,
            -54.9975,
            37.3003,
            234.1543,
            285.4766,
            220.4426,
            143.6304,
            137.5937,
            211.4197,
            262.0305,
            270.5154,
            389.1803,
            470.4571,
            525.2505,
            558.2873,
            591.8849,
            588.9648,
            446.0632,
            256.7899,
            201.0291,
            382.9406,
            617.7319,
            779.6067,
            832.6016,
            852.8168,
            843.9646,
            832.5167,
            783.9301,
            670.7439,
            559.9111,
            498.9460,
            491.1658,
            407.4325,
            438.6324,
            478.8776,
            421.8511,
            317.5572,
            179.2499,
            -30.8272,
            -284.5545,
            -432.8229,
            -493.2931,
            -513.7524,
        ]
    ),
    torch.tensor(
        [
            115.1510,
            215.3716,
            287.5760,
            288.7621,
            226.1282,
            179.6973,
            289.1445,
            358.4659,
            287.7536,
            124.4164,
            28.4071,
            141.3269,
            387.1749,
            531.5742,
            572.3959,
            669.6034,
            680.5667,
            714.7567,
            730.6467,
            808.1053,
            913.2842,
            834.1342,
            568.6379,
            395.8460,
            421.2122,
            562.9326,
            697.8311,
            889.1922,
            905.0083,
            722.0132,
            573.4169,
            633.7318,
            620.1314,
            437.8107,
            318.5802,
            319.9866,
            380.1544,
            337.4947,
            303.6475,
            338.7592,
            351.6160,
            357.8156,
            424.5802,
            432.9986,
            384.6477,
            332.5588,
            277.3048,
            292.1295,
            325.6743,
            354.7916,
            361.2131,
            432.3568,
            492.7018,
            558.8896,
            583.5618,
            537.9579,
            485.7822,
            375.1738,
            214.6532,
            92.5908,
            128.7328,
            255.0811,
            425.2848,
            442.7954,
            402.1526,
            368.0806,
            365.9096,
            330.9934,
            285.5682,
            405.1945,
            477.7957,
            481.2300,
            483.0048,
            478.2487,
            494.7870,
            535.8517,
            516.6946,
            506.5249,
            561.2392,
            749.8998,
            955.5564,
            1044.8842,
            1004.9779,
            799.0158,
            608.5074,
            502.1793,
            411.9709,
            348.6377,
            373.7067,
            430.5972,
            282.1737,
            192.5337,
            185.5119,
            143.7387,
            61.5814,
            10.3220,
            -57.3862,
            -169.2127,
            -273.4474,
            -161.8841,
            194.8448,
        ]
    ),
    torch.tensor(
        [
            776.1372,
            378.6538,
            246.0796,
            166.7325,
            96.2762,
            148.2516,
            184.3747,
            117.7323,
            6.1721,
            -101.7311,
            -128.3934,
            -240.9201,
            -502.3073,
            -678.8829,
            -619.5409,
            -556.2909,
            -507.9849,
            -462.0529,
            -429.2398,
            -467.1233,
            -506.4708,
            -462.2921,
            -292.9561,
            -108.8152,
            -43.9907,
            -3.6956,
            -20.1537,
            -206.5346,
            -244.2454,
            -67.3143,
            145.2849,
            119.3495,
            44.4147,
            63.5164,
            116.6548,
            108.4716,
            91.9740,
            62.4318,
            3.0266,
            -48.8548,
            -29.0962,
            177.7222,
            341.3843,
            310.8607,
            143.6693,
            87.9677,
            230.1508,
            203.7903,
            45.0270,
            -73.1238,
            -116.1827,
            -220.7386,
            -197.5807,
            -99.1505,
            -23.9602,
            25.7145,
            19.6036,
            196.7342,
            368.1356,
            402.7599,
            291.1914,
            151.3131,
            206.4843,
            313.8150,
            356.3717,
            337.3317,
            230.0277,
            239.6573,
            316.0160,
            205.8771,
            57.2944,
            -55.8089,
            -45.6667,
            -34.7787,
            88.2360,
            286.1979,
            442.9514,
            436.2472,
            242.6004,
            -41.1679,
            -236.4388,
            -276.7057,
            -310.4146,
            -316.4443,
            -360.5294,
            -404.6057,
            -246.5071,
            10.4858,
            161.5504,
            136.8478,
            159.1438,
            122.5914,
            167.7404,
            242.9027,
            265.1239,
            332.6238,
            528.4102,
            732.6014,
            895.4438,
            921.6850,
            729.0729,
        ]
    ),
    torch.tensor(
        [
            887.6834,
            416.8447,
            277.7835,
            262.1888,
            272.2485,
            303.2494,
            345.2531,
            376.2636,
            319.7996,
            198.1135,
            183.0794,
            109.4705,
            -106.9752,
            -352.7314,
            -377.8470,
            -472.6277,
            -497.8013,
            -410.4395,
            -350.8788,
            -466.2236,
            -552.9739,
            -460.9907,
            -276.6916,
            -147.0479,
            -57.5764,
            -19.4935,
            13.5391,
            -178.8044,
            -278.0436,
            -184.6884,
            12.4561,
            36.1375,
            72.4527,
            301.4344,
            432.3571,
            392.7298,
            270.6792,
            148.9332,
            -8.3913,
            -127.2065,
            -155.9837,
            -28.5532,
            112.4081,
            190.9356,
            157.9060,
            168.8748,
            255.8113,
            221.1170,
            187.5372,
            158.4972,
            75.6060,
            -59.0027,
            -17.5503,
            63.1098,
            28.2679,
            -114.3391,
            -215.9401,
            6.5071,
            381.4751,
            620.3768,
            575.8492,
            313.2881,
            167.9508,
            239.7618,
            348.6196,
            467.4425,
            484.8890,
            365.4383,
            205.3285,
            81.6230,
            159.6722,
            197.7296,
            123.9022,
            -42.2559,
            -101.8619,
            62.3644,
            276.7266,
            290.8541,
            111.5125,
            -11.1642,
            -101.4868,
            -177.3621,
            -339.8015,
            -461.8991,
            -488.6625,
            -501.0956,
            -405.8889,
            -211.3254,
            -81.2882,
            -123.8613,
            -91.9449,
            -124.6169,
            -174.2245,
            -118.8349,
            43.1947,
            287.3934,
            505.7382,
            750.3569,
            855.4404,
            947.9994,
            1001.5544,
        ]
    ),
    torch.tensor(
        [
            508.8242,
            259.2480,
            243.8576,
            299.3820,
            326.7904,
            399.7072,
            391.5456,
            332.4588,
            259.5930,
            212.1516,
            236.5614,
            129.4807,
            -114.1940,
            -379.6846,
            -418.5243,
            -445.3262,
            -437.7785,
            -385.3312,
            -330.1543,
            -376.1145,
            -375.6793,
            -285.6049,
            -245.1745,
            -335.0849,
            -374.5725,
            -259.6834,
            -25.9095,
            -31.6906,
            -106.9206,
            -84.0424,
            -19.6688,
            -20.4860,
            -50.7306,
            65.0592,
            170.7501,
            213.9359,
            190.2629,
            238.3477,
            248.6059,
            174.5655,
            184.2019,
            356.0968,
            397.3220,
            221.6722,
            15.2477,
            67.2684,
            201.0709,
            226.7585,
            142.1335,
            47.1041,
            29.2662,
            25.0131,
            38.5277,
            -35.5958,
            -176.6529,
            -273.5062,
            -237.7337,
            -6.7668,
            257.1884,
            432.6407,
            402.2936,
            346.7248,
            401.0662,
            445.7740,
            467.7643,
            479.9445,
            465.4988,
            486.2639,
            488.7386,
            339.3741,
            212.8381,
            89.2282,
            11.5005,
            -51.7300,
            -66.3455,
            152.7464,
            397.8792,
            502.7648,
            338.8419,
            147.2128,
            19.8165,
            -87.9850,
            -221.4004,
            -227.6935,
            -146.8167,
            -25.8282,
            28.2108,
            -10.9178,
            -15.0882,
            -16.9935,
            24.6959,
            13.6326,
            19.5662,
            70.2372,
            268.4411,
            556.8915,
            793.6505,
            913.0347,
            909.7551,
            990.3922,
            1122.3236,
        ]
    ),
]
