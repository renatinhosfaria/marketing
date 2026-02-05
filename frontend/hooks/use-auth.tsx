"use client";

import { createContext, useContext, type ReactNode } from "react";

export interface User {
  id: number;
  username: string;
  fullName: string;
  email: string;
  role: string;
  department: string;
  isActive: boolean;
}

interface AuthContextType {
  user: User;
  isLoading: boolean;
  isAuthenticated: boolean;
}

const STATIC_USER: User = {
  id: 1,
  username: "admin",
  fullName: "Administrador",
  email: "",
  role: "Gestor",
  department: "",
  isActive: true,
};

const AuthContext = createContext<AuthContextType>({
  user: STATIC_USER,
  isLoading: false,
  isAuthenticated: true,
});

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <AuthContext.Provider value={{ user: STATIC_USER, isLoading: false, isAuthenticated: true }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
