#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FASTLLM_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# === 配置 ===
NDK_PATH=${ANDROID_NDK_HOME:-${ANDROID_NDK:-$HOME/android-ndk-r29}}
SDK_PATH=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Android/Sdk}}
BUILD_TOOLS="$SDK_PATH/build-tools/34.0.0"
PLATFORM_JAR="$SDK_PATH/platforms/android-34/android.jar"
ABI=arm64-v8a
API_LEVEL=26
TOOLCHAIN="$NDK_PATH/build/cmake/android.toolchain.cmake"

BUILD_DIR="$SCRIPT_DIR/build"
APK_NAME="fastllm.apk"

# === 检查工具 ===
check() { [ -e "$1" ] || { echo "Error: $2 not found at $1"; exit 1; }; }
check "$NDK_PATH" "Android NDK"
check "$TOOLCHAIN" "NDK toolchain"
check "$BUILD_TOOLS/aapt2" "aapt2"
check "$BUILD_TOOLS/d8" "d8"
check "$BUILD_TOOLS/zipalign" "zipalign"
check "$BUILD_TOOLS/apksigner" "apksigner"
check "$PLATFORM_JAR" "android.jar"

echo "=== Fastllm APP Builder ==="
echo "NDK:  $NDK_PATH"
echo "SDK:  $SDK_PATH"
echo ""

if [ "${1:-}" = "--clean" ]; then
    echo "Clean build requested."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"/{native,classes,apk_staging/lib/$ABI,gen,dex}

# === Step 1: NDK 编译 libfastllm.so ===
echo "[1/6] Compiling native library..."
NATIVE_BUILD="$BUILD_DIR/native"
cmake -S "$SCRIPT_DIR/jni" -B "$NATIVE_BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=$ABI \
    -DANDROID_NATIVE_API_LEVEL=$API_LEVEL \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_TFACC=OFF -DPY_API=OFF

cmake --build "$NATIVE_BUILD" -j$(nproc)

cp "$NATIVE_BUILD/libfastllm.so" "$BUILD_DIR/apk_staging/lib/$ABI/"

# === Step 2: 编译资源 ===
echo "[2/6] Compiling resources..."
"$BUILD_TOOLS/aapt2" compile -o "$BUILD_DIR/res.zip" --dir "$SCRIPT_DIR/res"

# === Step 3: 链接资源 ===
echo "[3/6] Linking resources..."
"$BUILD_TOOLS/aapt2" link \
    -o "$BUILD_DIR/res.apk" \
    -I "$PLATFORM_JAR" \
    --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
    --java "$BUILD_DIR/gen" \
    --min-sdk-version $API_LEVEL \
    --target-sdk-version 34 \
    "$BUILD_DIR/res.zip"

# === Step 4: 编译 Java ===
echo "[4/6] Compiling Java..."
find "$SCRIPT_DIR/java" "$BUILD_DIR/gen" -name "*.java" > "$BUILD_DIR/sources.txt"

NEED_RECOMPILE_JAVA=0
while IFS= read -r src; do
    classfile="$BUILD_DIR/classes/$(echo "$src" | sed "s|$SCRIPT_DIR/java/||;s|$BUILD_DIR/gen/||;s|\.java$|.class|")"
    if [ ! -f "$classfile" ] || [ "$src" -nt "$classfile" ]; then
        NEED_RECOMPILE_JAVA=1
        break
    fi
done < "$BUILD_DIR/sources.txt"

if [ "$NEED_RECOMPILE_JAVA" -eq 1 ]; then
    echo "  Java sources changed, recompiling..."
    javac -source 1.8 -target 1.8 \
        -classpath "$PLATFORM_JAR" \
        -d "$BUILD_DIR/classes" \
        @"$BUILD_DIR/sources.txt" 2>&1
else
    echo "  Java sources unchanged, skipping."
fi

# === Step 5: dex ===
echo "[5/6] Creating dex..."
NEWEST_CLASS=$(find "$BUILD_DIR/classes" -name "*.class" -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
DEX_TIME=$(stat -c '%Y' "$BUILD_DIR/dex/classes.dex" 2>/dev/null || echo 0)

if [ -z "$NEWEST_CLASS" ] || [ "$(echo "$NEWEST_CLASS > $DEX_TIME" | bc)" -eq 1 ]; then
    echo "  Regenerating dex..."
    "$BUILD_TOOLS/d8" \
        --min-api $API_LEVEL \
        --output "$BUILD_DIR/dex" \
        $(find "$BUILD_DIR/classes" -name "*.class")
else
    echo "  Dex is up to date, skipping."
fi

# === Step 6: 组装 APK ===
echo "[6/6] Assembling APK..."

cp "$BUILD_DIR/res.apk" "$BUILD_DIR/unsigned.apk"

cd "$BUILD_DIR/dex" && zip -u "$BUILD_DIR/unsigned.apk" classes.dex && cd "$SCRIPT_DIR"

cd "$BUILD_DIR/apk_staging" && zip -r -u "$BUILD_DIR/unsigned.apk" lib/ && cd "$SCRIPT_DIR"

"$BUILD_TOOLS/zipalign" -f 4 "$BUILD_DIR/unsigned.apk" "$BUILD_DIR/aligned.apk"

KEYSTORE="$BUILD_DIR/debug.keystore"
if [ ! -f "$KEYSTORE" ]; then
    keytool -genkeypair -v -keystore "$KEYSTORE" -storepass android \
        -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 \
        -validity 10000 -dname "CN=Debug,O=Fastllm,C=CN" 2>/dev/null
fi
"$BUILD_TOOLS/apksigner" sign \
    --ks "$KEYSTORE" --ks-pass pass:android \
    --key-pass pass:android --ks-key-alias androiddebugkey \
    --out "$SCRIPT_DIR/$APK_NAME" "$BUILD_DIR/aligned.apk"

echo ""
echo "=== Build succeeded ==="
echo "APK: $SCRIPT_DIR/$APK_NAME"
echo ""
echo "Install: adb install $SCRIPT_DIR/$APK_NAME"
