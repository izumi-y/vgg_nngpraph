//
//  ViewController.swift
//  vgg_mpsnngraph_sample
//
//  Created by 泉　裕貴 on 2017/12/05.
//  Copyright © 2017年 izumi. All rights reserved.
//

import UIKit
import Metal
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
    
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var debugImageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var timeLabel: UILabel!
    
    
    var device: MTLDevice!
    var videoCapture: VideoCapture!
    var commandQueue: MTLCommandQueue!
    
    var vgg: VGG!
    
    
    
    var framesDone = 0
    var frameCapturingStartTime: CFTimeInterval = 0
    let semaphore = DispatchSemaphore(value: 2)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        predictionLabel.text = ""
        timeLabel.text = ""
        
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
        
        commandQueue = device.makeCommandQueue()
        
        vgg = VGG(commandQueue: commandQueue)
        
        setUpCamera()
        
        frameCapturingStartTime = CACurrentMediaTime()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }
    
    // MARK: - Initialization
    
    func setUpCamera() {
        videoCapture = VideoCapture(device: device)
        videoCapture.delegate = self
        videoCapture.fps = 50
        videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.vga640x480) { success in
            if success {
                // Add the video preview into the UI.
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                
                // Once everything is set up, we can start capturing live video.
                self.videoCapture.start()
            }
        }
    }
    
    // MARK: - UI stuff
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    
    // MARK: - Doing inference
    
    func predict(texture: MTLTexture) {
        vgg.predict(texture: texture) { result in
            DispatchQueue.main.async {
                self.show(predictions: result.predictions)
                
//                if let texture = result.debugTexture {
//                    self.debugImageView.image = UIImage.image(texture: texture)
//                }
                
//                let fps = self.measureFPS()
                self.timeLabel.text = String(format: "Elapsed %.5f seconds ", result.elapsed)
                
                self.semaphore.signal()
            }
        }
    }
    
    func measureFPS() -> Double {
        // Measure how many frames were actually delivered per second.
        framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
        let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            framesDone = 0
            frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFPSDelivered
    }
    
    
//    func show(predictions: String) {
//        var s: String = ""
//        s = predictions
//        predictionLabel.text = s
//    }
    
    
    
    func show(predictions: [VGG.Prediction]) {
        var s: [String] = []

        
        for (i, pred) in predictions.enumerated() {
            s.append(String(format: "%d: %@ (%3.2f%%)", i + 1, pred.label, pred.score * 100))
        }
        
        predictionLabel.text = s.joined(separator: "\n")
    }
}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
        // For debugging.
        predict(texture: loadTexture(named: "dog.jpg")!); return
        
        // The semaphore is necessary because the call to predict() does not block.
        // If we _would_ be blocking, then AVCapture will automatically drop frames
        // that come in while we're still busy. But since we don't block, all these
        // new frames get scheduled to run in the future and we end up with a backlog
        // of unprocessed frames. So we're using the semaphore to block if predict()
        // is already processing 2 frames, and we wait until the first of those is
        // done. Any new frames that come in during that time will simply be dropped.
        semaphore.wait()
        
//        if let texture = texture {
//            predict(texture: texture)
//        }
    }
}


