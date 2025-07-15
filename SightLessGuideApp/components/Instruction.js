// Instructions.js

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  Image,
  ScrollView,
  Pressable,
  Linking,
} from 'react-native';

const Instructions = ({navigation}) => {
  const handleHelpPress = () => {
    const emails = ['adhilsalam200@gmail.com','eldhosoorajgeorge04@gmail.com', 'adhensarageorge06@gmail.com','angelelizabethgeorge22@gmail.com' ]
    Linking.openURL(
      `mailto:${emails.concat(',')}?subject=Help!!!&body=Please provide details of your issue here.`,
    );
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: 'white'}}>
      {/* <Pressable
        style={[style.button, style.helpButton]}
        onPress={handleHelpPress}>
        <Text style={style.buttonText}>back</Text>
      </Pressable> */}
      <ScrollView contentContainerStyle={style.container}>
        <Image
          source={require('../components/Images/Img1.jpg')}
          style={style.instructionImage}
        />
        <Text style={style.header}>Instructions</Text>

        <View style={style.instructionContainer}>
          <Text style={style.instructionText}>
            <Text style={style.step}>Step 1: </Text>
            Open the app and navigate to the Login screen. Enter your username
            and add up to four contacts to your list.
          </Text>
        </View>

        <View style={style.instructionContainer}>
          <Text style={style.instructionText}>
            <Text style={style.step}>Step 2: </Text>
            After successful registration, you will be taken to the Object
            Detection screen. 
          </Text>
        </View>

        <View style={style.instructionContainer}>
          <Text style={style.instructionText}>
            <Text style={style.step}>Step 3: </Text>
            Navigate through the app using voice commands. You can make
            emergency calls to your contacts directly from any module.
          </Text>
        </View>

        <View style={style.instructionContainer}>
          <Text style={style.instructionText}>
            <Text style={style.step}>Step 4: </Text>
            The app includes features like object detection, text detection with
            a double-tap, emergency calls and a
            real-time navigation system.
          </Text>
        </View>

        <View style={style.buttonContainer}>
          <Pressable
            style={style.button}
            onPress={() => navigation.navigate('Start')}>
            <Text style={style.buttonText}>Next</Text>
          </Pressable>
          <Pressable
            style={[style.button, style.helpButton]}
            onPress={handleHelpPress}>
            <Text style={style.buttonText}>Help</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const style = StyleSheet.create({
  container: {
    padding: 20,
    alignItems: 'center',
  },
  header: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'black',
    marginBottom: 20,
    textAlign: 'center',
  },
  instructionContainer: {
    marginBottom: 20,
  },
  instructionImage: {
    height: 200,
    width: '100%',
    borderRadius: 10,
    marginBottom: 20,
  },
  instructionText: {
    fontSize: 16,
    color: 'black',
  },
  step: {
    fontWeight: 'bold',
    color: '#333',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginTop: 20,
  },
  button: {
    backgroundColor: 'black',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 10,
    alignItems: 'center',
    width: '45%',
  },
  helpButton: {
    backgroundColor: 'gray',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default Instructions;
