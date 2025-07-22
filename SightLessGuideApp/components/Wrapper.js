import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Animated,
    Easing,
    Image,
    SafeAreaView
} from 'react-native';
import Voice from '@react-native-voice/voice';

const Wrapper = ({ navigation }) => {
    const [listening, setListening] = useState(false);
    const [command, setCommand] = useState('');
    const [buttonScale] = useState(new Animated.Value(1));

    useEffect(() => {
        Voice.onSpeechResults = onSpeechResults;
    }, []);

    const handlePressIn = () => {
        Animated.timing(buttonScale, {
            toValue: 0.9,
            duration: 100,
            easing: Easing.out(Easing.quad),
            useNativeDriver: true,
        }).start();
    };

    const handlePressOut = () => {
        Animated.timing(buttonScale, {
            toValue: 1,
            duration: 100,
            easing: Easing.out(Easing.quad),
            useNativeDriver: true,
        }).start();
    };

    const startListening = async () => {
        try {
            setListening(true);
            await Voice.start('en-US');
        } catch (error) {
            console.error('Error starting voice recognition:', error);
        }
    };

    const stopListening = async () => {
        try {
            setListening(false);
            await Voice.stop();
        } catch (error) {
            console.error('Error stopping voice recognition:', error);
        }
    };

    const onSpeechResults = (e) => {
        console.log('speech results:', e);
            if (e.value && e.value.length > 0) {
              const spokenText = e.value[0].toLowerCase();
              if (spokenText === 'scanner') {
                // Tts.stop();
                navigation.navigate('Textscanner');
              }
              else if (spokenText === 'update') {
                // Tts.stop();
                navigation.navigate('Registration');
              }
              else if (spokenText === 'help') {
                // Tts.stop();
                navigation.navigate('Emergencycontact');
              }
              else if (spokenText === 'currency detection') {
                // Tts.stop();
                navigation.navigate('currency');
              }
              else if (spokenText === 'location') {
                // Tts.stop();
                navigation.navigate('Navigation');
              }
              else if (spokenText === 'weather') {
                // Tts.stop();
                navigation.navigate('LocationWeather');
              }
              else if (spokenText === 'detect') {
                // Tts.stop();
                navigation.navigate('Objectdetection', {mode: ''});
              }
              else if (spokenText.startsWith('add')) {
                // Tts.stop();
                const name = spokenText.replace('add', '').trim();
                navigation.navigate('Objectdetection', {mode: 'add-face', name});
              }
              else if (spokenText === 'find who') {
                // Tts.stop();
                navigation.navigate('Objectdetection', {mode: 'recognize'});
              }
              else {
                console.log('Unknown command:', spokenText);
                }
          };

        // if (e.value && e.value.length > 0) {
        //     const spokenCommand = e.value[0];
        //     setCommand(spokenCommand);
        //     console.log('Recognized command:', spokenCommand);

        //     // Navigate or perform actions based on the command
        //     if (spokenCommand.toLowerCase() === 'navigate') {
        //         navigation.navigate('Navigation');
        //     } else if (spokenCommand.toLowerCase() === 'help') {
        //         navigation.navigate('Help');
        //     } else {
        //         console.log('Unknown command:', spokenCommand);
        //     }
        // }
    };

    return (
        <SafeAreaView style={{flex: 1, backgroundColor: 'white'}}>
        <View style={styles.container}>
            <Text style={styles.title}>
                {listening ? 'Listening...' : 'Tap the microphone to speak'}
            </Text>
            <Animated.View style={{ transform: [{ scale: buttonScale }] }}>
                <TouchableOpacity
                    style={styles.microphoneButton}
                    onPressIn={handlePressIn}
                    onPressOut={handlePressOut}
                    onPress={listening ? stopListening : startListening}
                >
                    <Image
                        source={require('../images/microphone.png')} // Ensure you have a microphone icon in the assets folder
                        style={styles.microphoneIcon}
                    />
                </TouchableOpacity>
            </Animated.View>
            {command ? (
                <Text style={styles.commandText}>Command: {command}</Text>
            ) : null}
        </View>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#f5f5f5',
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 20,
        color: '#333',
    },
    microphoneButton: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: '#4CAF50',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
        elevation: 3,
    },
    microphoneIcon: {
        width: 50,
        height: 50,
    },
    commandText: {
        marginTop: 20,
        fontSize: 16,
        color: '#555',
    },
});

export default Wrapper;