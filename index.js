/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as blazeface from '@tensorflow-models/blazeface';
// import * as tf from "@tensorflow/tfjs"
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import '@tensorflow/tfjs-backend-webgl';
// tfjsWasm.setWasmPaths(
//   `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);
tf.setBackend('wasm').then(() => setupPage());
const stats = new Stats();
stats.showPanel(0);
document.body.prepend(stats.domElement);
const modelURL = 'model3/model.json';
let modelFace;
let model;
let ctx;
let videoWidth;
let videoHeight;
let video;
let canvas;
let start_button = document.querySelector("#start-record");
let snap_button = document.querySelector("#snap-button");
let stop_button = document.querySelector("#stop-record");
let download_link = document.querySelector("#download-video");
let message = document.querySelector('#message')
let faceMessage = document.querySelector('#faceMessage')
let timeMessage = document.querySelector('#timeMessage')
let camera_stream = null;
let media_recorder = null
let blobs_recorded = [];
let base64_recorded = []
let dataPixel
let distanceNoseToCenter;
let layer = document.getElementById('layer')
ctx_layer = layer.getContext('2d')
let face = document.getElementById('face')
ctx_face = face.getContext('2d')
const state = {
  backend: 'wasm',
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu'])
  .onChange(async (backend) => {
    await tf.setBackend(backend);
  });
start_button.addEventListener('click', function () {
  // set MIME type of recording as video/webm
  media_recorder = new MediaRecorder(camera_stream, { mimeType: 'video/webm' });

  // event : new recorded video blob available 
  media_recorder.addEventListener('dataavailable', function (e) {
    blobs_recorded.push(e.data);
  });

  // event : recording stopped & all blobs sent
  media_recorder.addEventListener('stop', function () {
    // create local object URL from the recorded video blobs
    let video_local = URL.createObjectURL(new Blob(blobs_recorded, { type: 'video/mp4' }));
    download_link.href = video_local;
  });

  // start recording with each recorded blob having 1 second video
  media_recorder.start(1000);
});
stop_button.addEventListener('click', function () {
  media_recorder.stop();
  console.log('blobs_recorded', blobs_recorded);
});
async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  camera_stream = stream
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  stats.begin();
  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = true;
  const predictions = await model.estimateFaces(video, returnTensors, flipHorizontal, annotateBoxes);
  // console.log('predictions', predictions)
  if (predictions.length > 1) {
    message.innerHTML = `Message: More than 1 face`
  }
  else if (predictions.length == 0) {
    message.innerHTML = `Message: No face`
  }
  else {
    const nose = predictions[0].landmarks[2];
    let center = [320, 240]
    distanceNoseToCenter = distance_2_point(nose, center)
    const right_ear = predictions[0].landmarks[4];
    const left_ear = predictions[0].landmarks[5];
    let bottomRight = predictions[0].bottomRight;
    let topLeft = predictions[0].topLeft;
    let cropWidth = topLeft[0] - bottomRight[0]
    let cropHeight = bottomRight[1] - topLeft[1]
    if (cropWidth > cropHeight) {
      cropHeight = cropWidth
    }
    else {
      cropWidth = cropHeight
    }
    ctx_face.canvas.width = 300
    ctx_face.canvas.height = 300
    let sx = 640 - bottomRight[0] - cropWidth - cropWidth * 0.3
    let sy = topLeft[1] - cropHeight * 0.5
    let sWidth = cropWidth * 1.6
    let sHeight = cropHeight * 1.6
    ctx_face.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, 256, 256)
    let pixel = ctx_face.getImageData(0, 0, 256, 256);
    dataPixel = pixel.data
    let left_distance = distance_2_point(left_ear, nose)
    let right_distance = distance_2_point(right_ear, nose)
    let ratio = left_distance / right_distance
    if (distanceNoseToCenter && distanceNoseToCenter < 100) {
      if (ratio > 2.2) {
        message.innerHTML = `Message: Turn your face left`
      }
      else if (ratio < 0.45) {
        message.innerHTML = `Message: Turn your face right`
      }
      else {
        message.innerHTML = `Message: Keep your face still`
      }
    }
    else {
      message.innerHTML = `Message: Please fill your face in shape`
    }

  }
  if (predictions.length > 0 || (distanceNoseToCenter && distanceNoseToCenter < 100)) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
        if (annotateBoxes) {
          predictions[i].landmarks = predictions[i].landmarks.arraySync();
        }
      }
      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
      ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
      ctx.fillRect(start[0], start[1], size[0], size[1]);
      if (annotateBoxes) {
        const landmarks = predictions[i].landmarks;
        ctx.fillStyle = 'blue';
        for (let j = 0; j < landmarks.length; j++) {
          const x = landmarks[j][0];
          const y = landmarks[j][1];
          ctx.fillRect(x, y, 5, 5);
        }
      }
    }
  }

  stats.end();

  requestAnimationFrame(renderPrediction);
};
const getLabelFace = async (data) => {
  let result = arrayToRgbArray(data)
  processedImage = await tf.tensor3d(result)
  const before = Date.now();
  const prediction = await modelFace.predict(tf.reshape(processedImage, shape = [-1, 256, 256, 3]));
  const after = Date.now();
  timeMessage.innerHTML = `Model processing time ${after - before}ms`
  const label = prediction.argMax(axis = 1).dataSync()[0];
  return label
}
let arrayToRgbArray = (data) => {
  let input = []
  for (let i = 0; i < 256; i++) {
    input.push([])
    for (let j = 0; j < 256; j++) {
      input[i].push([])
      input[i][j].push(data[(i * 256 + j) * 4])
      input[i][j].push(data[(i * 256 + j) * 4 + 1])
      input[i][j].push(data[(i * 256 + j) * 4 + 2])
    }
  }
  // console.log('input', input)
  return input
}
const distance_2_point = (point_1, point_2) => {
  let x1 = point_1[0]
  let y1 = point_1[1]
  let x2 = point_2[0]
  let y2 = point_2[1]
  let distance = Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
  return distance
}
const initModelFace = async () => {
  if (!modelFace) modelFace = await tf.loadGraphModel(modelURL);
}
const setupPage = async () => {
  message.innerHTML = `Message: Setting up camera`
  await setupCamera();
  message.innerHTML = `Message: Setting up tf backend`
  await tf.setBackend(state.backend);
  message.innerHTML = `Message: Setting up model`
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;
  canvas = document.getElementById('output');
  layer.width = videoWidth;
  layer.height = videoHeight;
  ctx_layer.lineWidth = 3;
  ctx_layer.fillStyle = "rgba(236, 236, 236, 0.6)";
  // ctx_layer.setLineDash([6, 8]);
  ctx_layer.beginPath();
  ctx_layer.ellipse(320, 240, 120, 150, 0, 0, Math.PI * 2);
  ctx_layer.rect(640, 0, -640, 480);
  ctx_layer.stroke();
  ctx_layer.fill();
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
  model = await blazeface.load();
  await initModelFace()
  renderPrediction();
  setInterval(async () => {
    let label = await getLabelFace(dataPixel)
    faceMessage.innerHTML = `Label: ${label} ${messageFace[label]} `
  }, 300)
};

// setupPage();
