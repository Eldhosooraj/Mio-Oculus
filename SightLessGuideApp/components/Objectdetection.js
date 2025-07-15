import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  PermissionsAndroid,
  Platform,
} from 'react-native';
import Voice from '@react-native-voice/voice';
import { useIsFocused } from '@react-navigation/native';
import axios from 'axios';
import Tts from 'react-native-tts';
import SQLite from 'react-native-sqlite-storage';
import {
  useCameraPermission,
  useMicrophonePermission,
  useCameraDevice,
  Camera,
} from 'react-native-vision-camera';

const db = SQLite.openDatabase(
  { name: 'contSDB.db', createFromLocation: '~conSDB.db' },
  () => console.log('Database opened successfully.'),
  error => console.error('Error opening database: ', error),
);

function Objectdetection({ navigation }) {
  const isFocused = useIsFocused();
  const [isListening, setIsListening] = useState(false);
  const listeningIntervalRef = useRef(null); // To store interval reference
  const photoIntervalRef = useRef(null); // To store photo-taking interval reference
  const cameraRef = useRef(null);
  const [extraCommand, setExtraCommand] = useState('');
  const {
    hasPermission: cameraPermission,
    requestPermission: requestCameraPermission,
  } = useCameraPermission();
  const {
    hasPermission: microphonePermission,
    requestPermission: requestMicrophonePermission,
  } = useMicrophonePermission();
  const cameraDevice = useCameraDevice('back');
  const [loading, setLoading] = useState(true);
  var check;

  useEffect(() => {
    console.log({ cameraPermission, microphonePermission });
    if (!cameraPermission) {
      requestCameraPermission();
    }
    if (!microphonePermission) {
      requestMicrophonePermission();
    }
  }, [cameraPermission, microphonePermission]);

  useEffect(() => {
    if (cameraPermission && microphonePermission) {
      setLoading(false);
    }
  }, [cameraPermission, microphonePermission]);

  const startListening = async () => {
    try {
      await Voice.start('en-GB');
      setIsListening(true);
    } catch (err) {
      console.error('Error starting voice recognition: ', err);
      setIsListening(false);
    }
  };

  const stopListening = async () => {
    try {
      await Voice.stop();
      await Voice.cancel();
      setIsListening(false);
    } catch (err) {
      console.error('Error stopping voice recognition: ', err);
    }
  };

  const onSpeechResults = e => {
    if (e.value && e.value.length > 0) {
      const spokenText = e.value[0].toLowerCase();
      if (spokenText === 'scanner') {
        Tts.stop();
        navigation.navigate('Textscanner');
      }
      else if (spokenText === 'help') {
        Tts.stop();
        navigation.navigate('Emergencycontact');
      }
      else if (spokenText === 'currency detection') {
        Tts.stop();
        navigation.navigate('currency');
      }
      else if (spokenText === 'location') {
        Tts.stop();
        navigation.navigate('Navigation');
      }
      else if (spokenText === 'weather') {
        Tts.stop();
        navigation.navigate('LocationWeather');
      }
      else if (spokenText.startsWith('add')) {
        if (photoIntervalRef.current) {
          clearInterval(photoIntervalRef.current);
          photoIntervalRef.current = null;
        }
        Tts.stop();
        const name = spokenText.replace('add', '').trim();
      addFace(name);
      }
      else if (spokenText === 'find who') {
        if (photoIntervalRef.current) {
          clearInterval(photoIntervalRef.current);
          photoIntervalRef.current = null;
        }
        Tts.stop();
        recognizeFace();
      }
      else {
        setExtraCommand(spokenText);
      }
    }
  };

  const onSpeechEnd = () => {
    setIsListening(false);
  };

  const addFace = async (name) => {
    setExtraCommand("");
    console.log("ADD FACE");
    Tts.speak('Stand still while we take your photo');
    // startListening();
    await setTimeout(() => {}, 500);
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        flash: 'off',
        enableShutterSound: false,
      });

      const imageUri = `file://${photo.path}`;
      Tts.speak('Photo taken');
      startListening();
      // setTimeout(() => {
      //   stopListening();
      //   if (extraCommand.length > 0) {
          Tts.speak('Ok, Adding as '+name)
          const formData = new FormData();
          formData.append('file', {
            uri: imageUri,
            name: 'image.jpg',
            type: 'image/jpg',
          });
          formData.append('name', name ?? "Unknown");
          try {
            const response = axios.post(
              'http://192.168.1.11:5000/add_face',
              formData,
              {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
              }
            ).then(response => {
              console.log(response.data);
              Tts.speak('Face added');
            });
          } catch (error) {
            console.error('Error uploading image:', JSON.stringify(error));
          }
        } else {
          Tts.speak('No name provided. Please try again later');
        }
      // }, 3000);
    // }
  };

  const recognizeFace = async () => {
    console.log("recognize??");
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        flash: 'off',
        enableShutterSound: false,
      });

      const imageUri = `file://${photo.path}`;
      console.log({ imageUri });
      const formData = new FormData();
      formData.append('file', {
        uri: imageUri,
        name: 'image.jpg',
        type: 'image/jpg',
      });

      try {
        const response = await axios.post(
          'http://192.168.1.11:5000/recognize_face',
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          }
        );
        console.log(response.data.message);
        const res = response.data.message
        // if (res !== check) {
        //   check=res;

          Tts.speak(res);
        // } else {
        //   console.log('Response is the same as the previous one. No need to speak.');
        // }
      } catch (error) {
        console.log('Error uploading image:', error);
      }
    }
  };

  const takePhoto = async () => {
    console.log("takephoto??");
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        flash: 'off',
        enableShutterSound: false,
      });

      const imageUri = `file://${photo.path}`;
      console.log({ imageUri });
      const formData = new FormData();
      formData.append('file', {
        uri: imageUri,
        name: 'image.jpg',
        type: 'image/jpg',
      });

      try {
        const response = await axios.post(
          'http://192.168.1.11:5000/predict',
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          }
        );
        const predictions = response.data.predictions.toString();
        const caption = response.data.caption;
        console.log(response.data);
        if (predictions || caption) {
          // check=res;

          Tts.speak(`Objects are ${predictions} and the sample caption is ${caption}`);
        } else {
          console.log(res, check)
          console.log('Response is the same as the previous one. No need to speak.');
        }
      } catch (error) {
        console.log('Error uploading image:', JSON.stringify(error));
      }
    }
  };

  useEffect(() => {
    const requestPermission = async () => {
      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: 'Permission to use microphone',
            message: 'We need your permission to use your microphone',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          },
        );

        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          console.warn('Microphone permission denied');
        }
      }
    };

    if (isFocused) {
      requestPermission();

      Voice.onSpeechResults = onSpeechResults;
      Voice.onSpeechEnd = onSpeechEnd;
      Voice.onSpeechError = e => console.error('Speech Error: ', e);

      startListening();

      listeningIntervalRef.current = setInterval(() => {
        startListening();
      }, 5000);

      photoIntervalRef.current = setInterval(() => {
        takePhoto();
      }, 10000);
    } else {
      if (listeningIntervalRef.current) {
        clearInterval(listeningIntervalRef.current);
        listeningIntervalRef.current = null;
      }
      if (photoIntervalRef.current) {
        clearInterval(photoIntervalRef.current);
        photoIntervalRef.current = null;
      }
      stopListening();
      Voice.destroy().then(Voice.removeAllListeners);
    }

    return () => {
      if (listeningIntervalRef.current) {
        clearInterval(listeningIntervalRef.current);
        listeningIntervalRef.current = null;
      }
      if (photoIntervalRef.current) {
        clearInterval(photoIntervalRef.current);
        photoIntervalRef.current = null;
      }
      stopListening();
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, [isFocused]);

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Object Detection</Text>
      {isListening ? <Text>Listening...</Text> : <Text>Not Listening</Text>}
      {loading ? (
        <Text>Loading...</Text>
      ) : !cameraPermission || !microphonePermission ? (
        <Text>
          Please grant camera and microphone permissions to use the app.
        </Text>
      ) : !cameraDevice ? (
        <Text>No camera device available.</Text>
      ) : (
        <Camera
          ref={cameraRef}
          photo={true}
          style={styles.camera}
          device={cameraDevice}
          isActive={true}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
});

export default Objectdetection;
