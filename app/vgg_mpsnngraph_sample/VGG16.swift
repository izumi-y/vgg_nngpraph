//
//  VGG16.swift
//  vgg_mpsnngraph_sample
//
//  Created by 泉　裕貴 on 2017/12/05.
//  Copyright © 2017年 izumi. All rights reserved.
//

import MetalPerformanceShaders
import QuartzCore

class VGG {
    public static let inputWidth = 224
    public static let inputHeight = 224
    
//    let meanImage = mean
    
    let labels = imagenetLabels
    
    typealias Prediction = (label: String, score: Float)
    
    struct Result {
        var predictions = [Prediction]()
        var debugTexture: MTLTexture?
        var elapsed: CFTimeInterval = 0
    }
    
//    struct Result {
//        var predictions: String = ""
//        var debugTexture: MTLTexture?
//        var elapsed: CFTimeInterval = 0
//    }
    
    let commandQueue: MTLCommandQueue
    let graph: MPSNNGraph
    
    public init(commandQueue: MTLCommandQueue) {
        self.commandQueue = commandQueue
        
        
        
        
        // Create a placeholder for the input image.
        // Note: YOLO expects the input pixels to be in the range 0-1. Our input
        // texture most likely has pixels with values 0-255. However, since its
        // pixel format is .unorm8 and the channel format for the graph's input
        // image is .float16, Metal will automatically convert the pixels to be
        // between 0 and 1.
        
        
        
        
        let inputImage = MPSNNImageNode(handle: nil)
        
        
        let scale = MPSNNBilinearScaleNode(source: inputImage,
                                           outputSize: MTLSize(width: VGG.inputWidth,
                                                               height: VGG.inputHeight,
                                                               depth: 3))
        
        
//        let scale = MPSNNLanczosScaleNode(source: inputImage,
//                                          outputSize: MTLSize(width: VGG.inputWidth,
//                                                              height: VGG.inputHeight,
//                                                              depth: 3))
        
        let mean = MPSCNNConvolutionNode(source: scale.resultImage,
                                         weights: DataSource("conv0", 1, 1, 3, 3, useReLU: false))
        
        let conv1 = MPSCNNConvolutionNode(source: mean.resultImage,
                                          weights: DataSource("conv1", 3, 3, 3, 64))
        
        let conv2 = MPSCNNConvolutionNode(source: conv1.resultImage,
                                          weights: DataSource("conv2", 3, 3, 64, 64))
        
        let pool1 = MPSCNNPoolingMaxNode(source: conv2.resultImage, filterSize: 2)
        
        let conv3 = MPSCNNConvolutionNode(source: pool1.resultImage,
                                          weights: DataSource("conv3", 3, 3, 64, 128))
        
        let conv4 = MPSCNNConvolutionNode(source: conv3.resultImage,
                                          weights: DataSource("conv4", 3, 3, 128, 128))
        
        let pool2 = MPSCNNPoolingMaxNode(source: conv4.resultImage, filterSize: 2)
        
        let conv5 = MPSCNNConvolutionNode(source: pool2.resultImage,
                                          weights: DataSource("conv5", 3, 3, 128, 256))
        
        let conv6 = MPSCNNConvolutionNode(source: conv5.resultImage,
                                          weights: DataSource("conv6", 3, 3, 256, 256))
        
        let conv7 = MPSCNNConvolutionNode(source: conv6.resultImage,
                                          weights: DataSource("conv7", 3, 3, 256, 256))
        
        let pool3 = MPSCNNPoolingMaxNode(source: conv7.resultImage, filterSize: 2)
        
        let conv8 = MPSCNNConvolutionNode(source: pool3.resultImage,
                                          weights: DataSource("conv8", 3, 3, 256, 512))
        
        let conv9 = MPSCNNConvolutionNode(source: conv8.resultImage,
                                          weights: DataSource("conv9", 3, 3, 512, 512))
        
        let conv10 = MPSCNNConvolutionNode(source: conv9.resultImage,
                                           weights: DataSource("conv10", 3, 3, 512, 512))
        
        let pool4 = MPSCNNPoolingMaxNode(source: conv10.resultImage, filterSize: 2)
        
        let conv11 = MPSCNNConvolutionNode(source: pool4.resultImage,
                                           weights: DataSource("conv11", 3, 3, 512, 512))
        
        let conv12 = MPSCNNConvolutionNode(source: conv11.resultImage,
                                           weights: DataSource("conv12", 3, 3, 512, 512))
        
        let conv13 = MPSCNNConvolutionNode(source: conv12.resultImage,
                                           weights: DataSource("conv13", 3, 3, 512, 512))
        
        let pool5 = MPSCNNPoolingMaxNode(source: conv13.resultImage, filterSize: 2)
        
        
        
        let fc1 = MPSCNNFullyConnectedNode(source: pool5.resultImage,
                                           weights: DataSource("fc1", 7, 7, 512, 4096))
        
        let fc2 = MPSCNNFullyConnectedNode(source: fc1.resultImage,
                                           weights: DataSource("fc2", 1, 1, 4096, 4096))
        
        let fc3 = MPSCNNFullyConnectedNode(source: fc2.resultImage,
                                           weights: DataSource("fc3", 1, 1, 4096, 1000))
     
        let softmax = MPSCNNSoftMaxNode(source: fc3.resultImage)
        
        if let graph = MPSNNGraph(device: commandQueue.device,
                                  resultImage: softmax.resultImage) {
            self.graph = graph
        } else {
            fatalError("Error: could not initialize graph")
        }
        
        // Enable extra debugging output.
        //graph.options = .verbose
        print(graph.debugDescription)
    }
    
    public func predict(texture: MTLTexture, completionHandler handler: @escaping (Result) -> Void) {
        let startTime = CACurrentMediaTime()
        let inputImage = MPSImage(texture: texture, featureChannels: 3)
        
        graph.executeAsync(withSourceImages: [inputImage]) { outputImage, error in
            var result = Result()
            if let image = outputImage {
//                print(image.toFloatArray())
//                print(image.width)
//                print(image.height)
//                print(image.featureChannels)
                
                result.predictions = self.top5Labels(prediction: image.toFloatArray())
            }
            
            result.elapsed = CACurrentMediaTime() - startTime
            handler(result)
        }
    }
    
    func top5Labels(prediction: [Float]) -> [Prediction] {
        
        let numclasses = 1000
        precondition(prediction.count == numclasses)
        
//        print(prediction)
        
        // Combine the predicted probabilities and their array indices into a new
        // list, then sort it from greatest probability to smallest. Finally, take
        // the top 5 items and convert them into strings.
        
        typealias tuple = (idx: Int, prob: Float)
//        print(prediction)
        return zip(0...1000, prediction)
            .sorted(by: { (a: tuple, b: tuple) -> Bool in a.prob > b.prob })
            .prefix(through: 4)
            .map({ (x: tuple) -> Prediction in (label: labels[x.idx], score: x.prob) })
    }

    
    
    

    // The weights (and bias terms) must be provided by a data source object.
    // This also returns an MPSCNNConvolutionDescriptor that has the kernel size,
    // number of channels, which activation function to use, etc.
    class DataSource: NSObject, MPSCNNConvolutionDataSource {
        let name: String
        let kernelWidth: Int
        let kernelHeight: Int
        let inputFeatureChannels: Int
        let outputFeatureChannels: Int
        let useReLU: Bool
        
        var data: Data?
        var data_w: Data?
        var data_b: Data?
        
        
        init(_ name: String, _ kernelWidth: Int, _ kernelHeight: Int,
             _ inputFeatureChannels: Int, _ outputFeatureChannels: Int,
             useReLU: Bool = true) {
            self.name = name
            self.kernelWidth = kernelWidth
            self.kernelHeight = kernelHeight
            self.inputFeatureChannels = inputFeatureChannels
            self.outputFeatureChannels = outputFeatureChannels
            self.useReLU = useReLU
        }
        
        func descriptor() -> MPSCNNConvolutionDescriptor {
            let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                                   kernelHeight: kernelHeight,
                                                   inputFeatureChannels: inputFeatureChannels,
                                                   outputFeatureChannels: outputFeatureChannels)
            if useReLU {
                desc.setNeuronType(.reLU, parameterA: 0, parameterB: 0)
                
                
            } else {
                desc.setNeuronType(.none, parameterA: 0, parameterB: 0)
            }
            return desc
        }
        
        func weights() -> UnsafeMutableRawPointer {
            return UnsafeMutableRawPointer(mutating: (data_w! as NSData).bytes)
        }
        
        func biasTerms() -> UnsafeMutablePointer<Float>? {
//            return nil
            return (UnsafeMutableRawPointer(mutating: (data_b! as NSData).bytes)).assumingMemoryBound(to:
                Float.self)
        }
        
        func load() -> Bool {
//            print("load parameters")
            if 1 == 1{
                if let url = Bundle.main.url(forResource: "\(name)_w", withExtension: "bin") {
                    do {
                        data_w = try Data(contentsOf: url)
                        print("load \(name) weights")
                    } catch {
                        print("Error: could not load \(url): \(error)")
                    }
                }else{
                    print("could not get \(name) weight url")
                }
                if let url = Bundle.main.url(forResource: "\(name)_b", withExtension: "bin") {
                    do {
                        data_b = try Data(contentsOf: url)
                        print("load \(name) bias")
                    } catch {
                        print("Error: could not load \(url): \(error)")
                    }
                }else{
                    print("could not get \(name) bias url")
                }
                return true
            }
        }
        
        func purge() {
            data_w = nil
            data_b = nil
        }
        
        func label() -> String? {
            return name
        }
        
        func dataType() -> MPSDataType {
            return .float32
        }
    }
    
    
}



