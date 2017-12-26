//
//  mean.metal
//  vgg_mpsnngraph_sample
//
//  Created by 泉　裕貴 on 2017/12/20.
//  Copyright © 2017年 izumi. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void adjust_mean_rgb(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    half4 inColor = inTexture.read(gid);
    half4 outColor = half4(inColor.z*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.x*255.0h - 123.68h, 0.0h);
    outTexture.write(outColor, gid);
}
