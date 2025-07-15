import AsyncStorage from '@react-native-async-storage/async-storage';
import { useState, useEffect } from 'react';

const useAuth = () => {
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const [emergencyContact, setEmergencyContact] = useState<Array<any>>();

  useEffect(() => {
    const loadLoginState = async () => {
      const storedLoginState = await AsyncStorage.getItem('isLoggedIn');
      setIsLoggedIn(storedLoginState === 'true');
      const emergencyContactList = await AsyncStorage.getItem('emergencyContact');
      setEmergencyContact(emergencyContactList ? JSON.parse(emergencyContactList) : []);
    };
    loadLoginState();
  }, []);

  const login = async () => {
    await AsyncStorage.setItem('isLoggedIn', 'true');
    setIsLoggedIn(true);
  };

  const logout = async () => {
    await AsyncStorage.removeItem('isLoggedIn');
    setIsLoggedIn(false);
  };

  interface EmergencyContact {
    name: string;
    number: string;
  }

  const addEmergencyContact = async (newEmergencyContact: EmergencyContact[]): Promise<void> => {
    const updatedContacts = [...(emergencyContact || []), ...newEmergencyContact];
    await AsyncStorage.setItem('emergencyContact', JSON.stringify(updatedContacts));
    setEmergencyContact(updatedContacts);
    if(!isLoggedIn) {
      login();  
    }
  };

  return { isLoggedIn, login, logout, addEmergencyContact, emergencyContact };
};

export default useAuth;
