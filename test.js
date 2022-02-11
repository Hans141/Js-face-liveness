const tf = require("@tensorflow/tfjs");

tf.loadGraphModel("http://0.0.0.0:3000/model.json").then((model) => {
  console.log("hello", model);
});