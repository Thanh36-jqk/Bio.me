'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import axios from 'axios';
import Webcam from 'react-webcam';
import { Camera, Upload, Check, X, User, Mail, Calendar } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function RegisterPage() {
    const [step, setStep] = useState(0); // 0: info, 1: face, 2: iris, 3: fingerprint, 4: success
    const [formData, setFormData] = useState({
        name: '',
        age: '',
        email: ''
    });

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Webcam refs
    const webcamRef = useRef<Webcam>(null);
    const [capturedImages, setCapturedImages] = useState<string[]>([]);
    const [isCapturing, setIsCapturing] = useState(false);

    // File states for other biometrics
    const [irisFiles, setIrisFiles] = useState<File[]>([]);
    const [fpFiles, setFpFiles] = useState<File[]>([]);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleInfoSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (!formData.email || !formData.name || !formData.age) {
            setError('Please fill in all fields');
            return;
        }

        setLoading(true);

        try {
            // Register user info first
            const response = await axios.post(`${API_BASE}/register/user`, {
                name: formData.name,
                age: parseInt(formData.age),
                email: formData.email
            }, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });

            if (response.data.success) {
                setStep(1); // Move to Face capture
            } else {
                setError(response.data.message || 'Registration failed');
            }
        } catch (err: any) {
            setError(err.response?.data?.message || 'Error connecting to server');
        } finally {
            setLoading(false);
        }
    };

    const captureFace = useCallback(() => {
        if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                setCapturedImages(prev => [...prev, imageSrc]);
            }
        }
    }, [webcamRef]);

    const handleFaceSubmit = async () => {
        if (capturedImages.length < 5) {
            setError('Please capture at least 5 photos');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const formDataUpload = new FormData();
            formDataUpload.append('email', formData.email);

            // Convert base64 to blobs
            for (let i = 0; i < capturedImages.length; i++) {
                const fetchRes = await fetch(capturedImages[i]);
                const blob = await fetchRes.blob();
                formDataUpload.append('images', blob, `face_${i}.jpg`);
            }

            const response = await axios.post(`${API_BASE}/register/face`, formDataUpload, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                setStep(2); // Move to Iris
            }
        } catch (err: any) {
            setError('Failed to upload face images');
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (files: FileList | null, type: 'iris' | 'fingerprint') => {
        if (!files) return;
        const fileArray = Array.from(files);
        if (type === 'iris') setIrisFiles(prev => [...prev, ...fileArray]);
        else setFpFiles(prev => [...prev, ...fileArray]);
    };

    const handleBiometricSubmit = async (type: 'iris' | 'fingerprint') => {
        const files = type === 'iris' ? irisFiles : fpFiles;
        if (files.length < 3) {
            setError(`Please upload at least 3 ${type} images`);
            return;
        }

        setLoading(true);
        setError('');

        try {
            const formDataUpload = new FormData();
            formDataUpload.append('email', formData.email);
            files.forEach(file => formDataUpload.append('images', file));

            const response = await axios.post(`${API_BASE}/register/${type}`, formDataUpload, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                if (type === 'iris') setStep(3);
                else setStep(4); // Success
            }
        } catch (err: any) {
            setError(`Failed to register ${type}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen relative overflow-hidden bg-[#0A0A0A] text-white selection:bg-purple-500/30">
            {/* Background Effects */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-purple-600/10 blur-[120px]" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-blue-600/10 blur-[120px]" />
            </div>

            <div className="container mx-auto px-4 py-8">
                <div className="mb-8">
                    <Link href="/" className="text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                        ← Back to Home
                    </Link>
                </div>

                <div className="max-w-2xl mx-auto">
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 mb-4">
                            Create Account
                        </h1>
                        <p className="text-gray-400 text-lg">
                            Register with Multi-Factor Biometrics
                        </p>
                    </div>

                    {/* Progress Steps */}
                    <div className="flex justify-between mb-12 relative">
                        <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gray-800 -z-10" />
                        {['Info', 'Face', 'Iris', 'Fingerprint'].map((label, i) => (
                            <div key={label} className={`flex flex-col items-center gap-2 bg-[#0A0A0A] px-2 ${i <= step ? 'text-purple-400' : 'text-gray-600'}`}>
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all ${i <= step ? 'border-purple-500 bg-purple-500/20' : 'border-gray-700 bg-gray-900'
                                    }`}>
                                    {i < step ? <Check size={14} /> : i + 1}
                                </div>
                                <span className="text-sm font-medium">{label}</span>
                            </div>
                        ))}
                    </div>

                    <div className="bg-gray-900/50 backdrop-blur-xl border border-white/5 rounded-2xl p-8 shadow-2xl">
                        {error && (
                            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-2">
                                <X size={18} />
                                {error}
                            </div>
                        )}

                        {step === 0 && (
                            <form onSubmit={handleInfoSubmit} className="space-y-6">
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-400 mb-2">Full Name</label>
                                        <div className="relative">
                                            <User className="absolute left-4 top-3.5 text-gray-500" size={20} />
                                            <input
                                                type="text"
                                                name="name"
                                                value={formData.name}
                                                onChange={handleInputChange}
                                                className="w-full bg-gray-950 border border-gray-800 rounded-xl py-3 pl-12 pr-4 text-white focus:outline-none focus:border-purple-500 transition-colors"
                                                placeholder="John Doe"
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-400 mb-2">Age</label>
                                        <div className="relative">
                                            <Calendar className="absolute left-4 top-3.5 text-gray-500" size={20} />
                                            <input
                                                type="number"
                                                name="age"
                                                value={formData.age}
                                                onChange={handleInputChange}
                                                className="w-full bg-gray-950 border border-gray-800 rounded-xl py-3 pl-12 pr-4 text-white focus:outline-none focus:border-purple-500 transition-colors"
                                                placeholder="25"
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-400 mb-2">Email Address</label>
                                        <div className="relative">
                                            <Mail className="absolute left-4 top-3.5 text-gray-500" size={20} />
                                            <input
                                                type="email"
                                                name="email"
                                                value={formData.email}
                                                onChange={handleInputChange}
                                                className="w-full bg-gray-950 border border-gray-800 rounded-xl py-3 pl-12 pr-4 text-white focus:outline-none focus:border-purple-500 transition-colors"
                                                placeholder="john@example.com"
                                            />
                                        </div>
                                    </div>
                                </div>
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="w-full bg-purple-600 hover:bg-purple-500 text-white font-medium py-4 rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                                >
                                    {loading ? 'Processing...' : 'Continue to Face Scan →'}
                                </button>
                            </form>
                        )}

                        {step === 1 && (
                            <div className="space-y-6">
                                <div className="text-center">
                                    <h3 className="text-xl font-bold text-white mb-2">Face Registration</h3>
                                    <p className="text-gray-400">Position your face in the camera and capture 5 photos</p>
                                </div>

                                <div className="relative aspect-video bg-black rounded-2xl overflow-hidden border border-gray-800">
                                    <Webcam
                                        audio={false}
                                        ref={webcamRef}
                                        screenshotFormat="image/jpeg"
                                        className="w-full h-full object-cover"
                                    />
                                </div>

                                <div className="grid grid-cols-5 gap-2">
                                    {[...Array(5)].map((_, i) => (
                                        <div key={i} className={`aspect-square rounded-lg border-2 overflow-hidden ${i < capturedImages.length ? 'border-purple-500' : 'border-gray-800 bg-gray-900'
                                            }`}>
                                            {capturedImages[i] && (
                                                <img src={capturedImages[i]} alt={`Captured ${i + 1}`} className="w-full h-full object-cover" />
                                            )}
                                        </div>
                                    ))}
                                </div>

                                <div className="flex gap-4">
                                    <button
                                        onClick={() => setCapturedImages([])}
                                        className="flex-1 px-6 py-4 rounded-xl border border-gray-700 text-gray-300 hover:bg-gray-800 transition-all"
                                    >
                                        Retake All
                                    </button>
                                    <button
                                        onClick={captureFace}
                                        disabled={capturedImages.length >= 5}
                                        className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-medium py-4 rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                                    >
                                        <Camera size={20} />
                                        Capture ({capturedImages.length}/5)
                                    </button>
                                </div>

                                <button
                                    onClick={handleFaceSubmit}
                                    disabled={capturedImages.length < 5 || loading}
                                    className="w-full bg-purple-600 hover:bg-purple-500 text-white font-medium py-4 rounded-xl transition-all disabled:opacity-50"
                                >
                                    {loading ? 'Uploading...' : 'Continue to Iris Scan →'}
                                </button>
                            </div>
                        )}

                        {(step === 2 || step === 3) && (
                            <div className="space-y-6">
                                <div className="text-center">
                                    <h3 className="text-xl font-bold text-white mb-2">
                                        {step === 2 ? 'Iris' : 'Fingerprint'} Registration
                                    </h3>
                                    <p className="text-gray-400">Upload at least 3 clear images</p>
                                </div>

                                <div className="border-2 border-dashed border-gray-700 rounded-2xl p-12 text-center hover:border-purple-500/50 hover:bg-purple-500/5 transition-all">
                                    <input
                                        type="file"
                                        multiple
                                        accept="image/*"
                                        onChange={(e) => handleFileUpload(e.target.files, step === 2 ? 'iris' : 'fingerprint')}
                                        className="hidden"
                                        id="file-upload"
                                    />
                                    <label htmlFor="file-upload" className="cursor-pointer block">
                                        <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mx-auto mb-4">
                                            <Upload className="text-purple-400" size={32} />
                                        </div>
                                        <p className="text-white font-medium mb-2">Click to Upload</p>
                                        <p className="text-gray-500 text-sm">JPG, PNG supported</p>
                                    </label>
                                </div>

                                <div className="text-center text-sm text-gray-400">
                                    Selected: {(step === 2 ? irisFiles : fpFiles).length} files
                                </div>

                                <button
                                    onClick={() => handleBiometricSubmit(step === 2 ? 'iris' : 'fingerprint')}
                                    disabled={(step === 2 ? irisFiles : fpFiles).length < 3 || loading}
                                    className="w-full bg-purple-600 hover:bg-purple-500 text-white font-medium py-4 rounded-xl transition-all disabled:opacity-50"
                                >
                                    {loading ? 'Processing...' : step === 2 ? 'Continue to Fingerprint →' : 'Complete Registration'}
                                </button>
                            </div>
                        )}

                        {step === 4 && (
                            <div className="text-center py-12">
                                <div className="w-24 h-24 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Check className="text-green-500" size={48} />
                                </div>
                                <h2 className="text-3xl font-bold text-white mb-4">Registration Complete!</h2>
                                <p className="text-gray-400 mb-8">
                                    Your biometric profile has been successfully created.
                                </p>
                                <Link href="/login" className="inline-block bg-white text-black font-bold py-4 px-12 rounded-xl hover:bg-gray-200 transition-colors">
                                    Go to Login
                                </Link>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
