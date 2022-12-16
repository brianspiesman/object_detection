import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { Button, Image, StyleSheet, Text, View } from "react-native";
import Colors from "../constants/Colors";
import * as ImagePicker from "expo-image-picker";
import { decodeJpeg, bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";

const ImgPicker = (props) => {
  const [pickedImage, setPickedImage] = useState();
  const [displayText, setDisplayText] = useState("Loading Vision Model");
  const [model, setModel] = useState(null);

  useEffect(() => {
    (async () => {
      await tf.ready();

      const modelJson = require("../assets/VisModels/model_OD_yolov5n640.json");
      const modelWeights1 = require("../assets/VisModels/model_OD_yolov5n640_weights-shard1of2.bin");
      const modelWeights2 = require("../assets/VisModels/model_OD_yolov5n640_weights-shard2of2.bin");

      const model = await tf.loadGraphModel(
        bundleResourceIO(modelJson, [modelWeights1, modelWeights2])
      );
      setModel(model);
      setDisplayText("Vision Model Ready!");
    })();
  }, []);

  useEffect(() => {
    (async () => {
      if (Platform.OS !== "web") {
        const { status } =
          await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== "granted") {
          alert("Sorry, BeeMachine needs camera roll permissions!");
        }
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== "granted") {
        alert("Sorry, BeeMachine needs camera permissions!");
      }
    })();
  }, []);

  async function getPrediction() {
    const result = await ImagePicker.launchImageLibraryAsync({
      quality: 1,
      exif: true,
    });

    if (!result.canceled) {
      setPickedImage(result.assets[0].uri);
    }
    props.onImageTaken(result.assets[0].uri);
    setDisplayText("Processing image");

    const fileUri = result.assets[0].uri;
    const IMGSIZE = 640;

    const imgB64 = await FileSystem.readAsStringAsync(fileUri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const imgBuffer = tf.util.encodeString(imgB64, "base64").buffer;
    const imageData = new Uint8Array(imgBuffer);
    const imageTensor = decodeJpeg(imageData)
      .expandDims()
      .resizeBilinear([IMGSIZE, IMGSIZE])
      .div(tf.scalar(255))
      .reshape([1, IMGSIZE, IMGSIZE, 3]);

    setDisplayText("Locating bee");

    const prediction = await model.executeAsync(imageTensor);

    const boxLocation = prediction[0].dataSync();
    const xmin = boxLocation[0];
    const ymin = boxLocation[1];
    const xmax = boxLocation[2];
    const ymax = boxLocation[3];

    const confidence = prediction[1].dataSync();
    const objClass = prediction[2].dataSync();
    const numDetections = prediction[3].dataSync();

    console.log("*************");
    console.log("Xmin: ", xmin);
    console.log("Ymin: ", ymin);
    console.log("Xmax: ", xmax);
    console.log("Ymax: ", ymax);
    console.log("confidence: ", confidence[0]);
    console.log("class: ", objClass[0]);
    console.log("number detections: ", numDetections[0]);
    console.log("*************");

    setDisplayText("Done");
  }

  return (
    <View style={styles.imagePicker}>
      <Text style={styles.dispText}>{displayText}</Text>
      <View style={styles.imagePreview}>
        {!pickedImage ? (
          <Text>No image picked yet.</Text>
        ) : (
          <Image style={styles.image} source={{ uri: pickedImage }} />
        )}
      </View>
      <View style={styles.buttonContainer}>
        <Button
          title="Get Image"
          color={Colors.primary}
          onPress={getPrediction}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  buttonContainer: {
    justifyContent: "space-evenly",
    flexDirection: "row",
    width: "100%",
  },
  imagePicker: {
    alignItems: "center",
    marginBottom: 5,
  },
  imagePreview: {
    width: "100%",
    height: 250,
    marginBottom: 10,
    justifyContent: "center",
    alignItems: "center",
    borderColor: "#ccc",
    borderWidth: 1,
  },
  image: {
    resizeMode: "contain",
    width: "100%",
    height: "100%",
  },
});

export default ImgPicker;
