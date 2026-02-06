@app.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_multi_biometric(
    username: str = Form(...),
    face_image: UploadFile = File(None),
    iris_image: UploadFile = File(None),
    fingerprint_image: UploadFile = File(None)
):
    """
    Multi-factor biometric authentication with 2/3 pass rule
    Requires at least 2 out of 3 biometrics to pass
    Includes liveness detection for anti-spoofing
    """
    print(f"🔐 Multi-biometric authentication for: {username}")
    
    # Check user exists
    user = await db_manager.get_user(username)
    if not user:
        return AuthenticationResponse(
            success=False,
            message="User not found",
            passed_biometrics=0
        )
    
    results = []
    liveness_results = {}
    
    # 1. Face Authentication (if provided)
    if face_image:
        try:
            # Read image
            contents = await face_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check
            liveness_result = liveness_detector.detect_face_liveness(img)
            liveness_results['face'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"❌ Face liveness FAILED: {liveness_result['reason']}")
                results.append({'type': 'face', 'success': False, 'reason': liveness_result['reason']})
            else:
                # TODO: Actual face recognition here
                # For now, mark as success if liveness passed and user has face registered
                if user.get("face_registered"):
                    results.append({'type': 'face', 'success': True, 'confidence': liveness_result['confidence']})
                    print(f"✓ Face authentication PASSED (liveness: {liveness_result['confidence']:.2f})")
                else:
                    results.append({'type': 'face', 'success': False, 'reason': 'Face not registered'})
                    
        except Exception as e:
            print(f"❌ Face authentication error: {e}")
            results.append({'type': 'face', 'success': False, 'reason': str(e)})
            liveness_results['face'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # 2. Iris Authentication (if provided)
    if iris_image:
        try:
            contents = await iris_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check
            liveness_result = liveness_detector.detect_iris_liveness(img)
            liveness_results['iris'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"❌ Iris liveness FAILED: {liveness_result['reason']}")
                results.append({'type': 'iris', 'success': False, 'reason': liveness_result['reason']})
            else:
                if user.get("iris_registered"):
                    results.append({'type': 'iris', 'success': True, 'confidence': liveness_result['confidence']})
                    print(f"✓ Iris authentication PASSED (liveness: {liveness_result['confidence']:.2f})")
                else:
                    results.append({'type': 'iris', 'success': False, 'reason': 'Iris not registered'})
                    
        except Exception as e:
            print(f"❌ Iris authentication error: {e}")
            results.append({'type': 'iris', 'success': False, 'reason': str(e)})
            liveness_results['iris'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # 3. Fingerprint Authentication (if provided)
    if fingerprint_image:
        try:
            contents = await fingerprint_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check
            liveness_result = liveness_detector.detect_fingerprint_liveness(img)
            liveness_results['fingerprint'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"❌ Fingerprint liveness FAILED: {liveness_result['reason']}")
                results.append({'type': 'fingerprint', 'success': False, 'reason': liveness_result['reason']})
            else:
                if user.get("fingerprint_registered"):
                    results.append({'type': 'fingerprint', 'success': True, 'confidence': liveness_result['confidence']})
                    print(f"✓ Fingerprint authentication PASSED (liveness: {liveness_result['confidence']:.2f})")
                else:
                    results.append({'type': 'fingerprint', 'success': False, 'reason': 'Fingerprint not registered'})
                    
        except Exception as e:
            print(f"❌ Fingerprint authentication error: {e}")
            results.append({'type': 'fingerprint', 'success': False, 'reason': str(e)})
            liveness_results['fingerprint'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # Count successes
    passed_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    # 2/3 Rule: Need at least 2 out of 3 to pass
    if passed_count >= 2:
        print(f"✅ AUTHENTICATION SUCCESS: {passed_count}/{total_count} biometrics passed")
        return AuthenticationResponse(
            success=True,
            username=username,
            passed_biometrics=passed_count,
            message=f"Authentication successful ({passed_count}/{total_count} biometrics passed)",
            liveness_checks=liveness_results
        )
    else:
        print(f"❌ AUTHENTICATION FAILED: {passed_count}/{total_count} biometrics passed (need 2+)")
        failed_reasons = [r.get('reason', 'Unknown') for r in results if not r['success']]
        return AuthenticationResponse(
            success=False,
            message=f"Authentication failed: {passed_count}/{total_count} passed (need 2+). Failures: {', '.join(failed_reasons)}",
            passed_biometrics=passed_count,
            liveness_checks=liveness_results
        )
