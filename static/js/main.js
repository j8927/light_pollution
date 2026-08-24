// AI 분석 흐름 및 페이지 전환
const sampleImages = [
  { url: "/static/assets/images/samples/night_sign_01.jpg", name: "야간 간판 실사 1" },
  { url: "/static/assets/images/samples/night_sign_02.jpg", name: "전광판 실사 2" },
  { url: "/static/assets/images/samples/night_sign_03.jpg", name: "네온 간판 실사 3" },
  { url: "/static/assets/images/samples/night_sign_04.jpg", name: "도심 광고판 실사 4" },
  { url: "/static/assets/images/samples/night_sign_05.jpg", name: "가로등/간판 실사 5" },
  { url: "/static/assets/images/samples/night_sign_06.jpg", name: "상점 간판 실사 6" },
  { url: "/static/assets/images/samples/night_sign_07.jpg", name: "LED 전광판 실사 7" },
  { url: "/static/assets/images/samples/night_sign_08.jpg", name: "야간 조명 간판 실사 8" }
];

const CONTACT_INFO = {
  leader: { name: "조태승 (팀장)", email: "2143412@donga.ac.kr", phone: "010-8603-8271" },
  members: [
    { name: "곽승우", email: "2353660@donga.ac.kr" },
    { name: "김동규", email: "2353695@donga.ac.kr" },
    { name: "김승주", email: "2353716@donga.ac.kr" }
  ],
  address: "동아대학교 승학캠퍼스"
};

const CAMERA_MODEL_LABELS = {
  smartphone: "스마트폰 기본",
  iphone: "아이폰 계열",
  galaxy: "갤럭시 계열",
  dslr: "디지털카메라/DSLR",
  action: "액션캠/드론",
  cctv: "CCTV/고정형",
  default: "기본값"
};

const PAGE_VERSION = "20260511-1";

function withVersion(path) {
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}v=${PAGE_VERSION}`;
}

function syncFooterContact() {
  const footerTitles = document.querySelectorAll(".site-footer .footer-title");
  let contactContainer = null;
  footerTitles.forEach((titleEl) => {
    if (titleEl.textContent.trim() === "문의") {
      contactContainer = titleEl.parentElement;
    }
  });
  if (!contactContainer) return;

  // Keep footer contact fixed even if a stale HTML copy is loaded.
  contactContainer.innerHTML = [
    '<p class="footer-title">문의</p>',
    `<p>${CONTACT_INFO.leader.name}: ${CONTACT_INFO.leader.email}</p>`,
    `<p>${CONTACT_INFO.leader.phone}</p>`,
    ...CONTACT_INFO.members.map(m => `<p>${m.name}: ${m.email}</p>`),
    `<p>${CONTACT_INFO.address}</p>`
  ].join("");
}

function getRandomSampleImage() {
  return sampleImages[Math.floor(Math.random() * sampleImages.length)];
}

function readCaptureSettings() {
  const cameraModel = sessionStorage.getItem("light_cameraModel") || "smartphone";
  const iso = Number(sessionStorage.getItem("light_cameraIso") || "100");
  const exposureMs = Number(sessionStorage.getItem("light_cameraExposure") || "12");
  const angleDeg = Number(sessionStorage.getItem("light_cameraAngle") || "0");
  const locationMode = sessionStorage.getItem("light_locationMode") || "auto";
  const locationZone = sessionStorage.getItem("light_locationZone") || "제3종";
  
  return {
    cameraModel,
    cameraLabel: CAMERA_MODEL_LABELS[cameraModel] || CAMERA_MODEL_LABELS.default,
    iso: Number.isFinite(iso) ? iso : 100,
    exposureMs: Number.isFinite(exposureMs) ? exposureMs : 12,
    angleDeg: Number.isFinite(angleDeg) ? angleDeg : 0,
    locationMode,
    locationZone,
  };
}

function saveCaptureSettings() {
  const capture = readCaptureSettings();
  sessionStorage.setItem("light_cameraModel", capture.cameraModel);
  sessionStorage.setItem("light_cameraLabel", capture.cameraLabel);
  sessionStorage.setItem("light_cameraIso", String(capture.iso));
  sessionStorage.setItem("light_cameraExposure", String(capture.exposureMs));
  sessionStorage.setItem("light_cameraAngle", String(capture.angleDeg));
  sessionStorage.setItem("light_locationMode", capture.locationMode);
  sessionStorage.setItem("light_locationZone", capture.locationZone);
  return capture;
}

function formatCaptureSummary(capture) {
  if (!capture) return "-";
  const cameraLabel = capture.cameraLabel || CAMERA_MODEL_LABELS[capture.cameraModel] || capture.cameraModel || "-";
  const zoneMap = {
    '제1종': '제1종 자연환경보존지역',
    '제2종': '제2종 농림지역',
    '제3종': '제3종 주거지역',
    '제4종': '제4종 상업공업지역'
  };
  const locationText = `위치: ${zoneMap[capture.locationZone] || capture.locationZone || '제3종 주거지역'}`;
  return `${cameraLabel} · ISO ${capture.iso} · ${capture.exposureMs}ms · ${capture.angleDeg}° · ${locationText}`;
}

/**
 * JPEG ArrayBuffer에서 EXIF 카메라 모델명을 파싱합니다.
 * 추출된 모델명을 알려진 카메라 기종으로 자동 매핑합니다.
 * @returns {{model: string, cameraKey: string}|null} model: 원본 모델명, cameraKey: 매핑된 기종 키
 */
function extractCameraModelFromArrayBuffer(buffer) {
  try {
    const dv = new DataView(buffer);
    // JPEG 시그니처 확인
    if (dv.getUint16(0, false) !== 0xFFD8) return null;

    let offset = 2;
    let tiffStart = -1;
    let isLittleEndian = false;
    let exifIfdOffset = -1;
    let model = null;
    let cameraKey = 'smartphone';
    let iso = 100;
    let exposureMs = 12;

    function bytesPerComponent(format) {
      return ({ 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 }[format] || 1);
    }

    function readEntryValue(entryOffset, format, components) {
      const size = bytesPerComponent(format) * Math.max(1, components || 1);
      const valueOffset = size <= 4 ? entryOffset + 8 : tiffStart + dv.getUint32(entryOffset + 8, isLittleEndian);
      if (format === 2) {
        const bytes = new Uint8Array(buffer, valueOffset, Math.min(components || 0, 128));
        return String.fromCharCode(...bytes).split('\0')[0].trim();
      }
      if (format === 3) return dv.getUint16(valueOffset, isLittleEndian);
      if (format === 4) return dv.getUint32(valueOffset, isLittleEndian);
      if (format === 5) {
        const num = dv.getUint32(valueOffset, isLittleEndian);
        const den = dv.getUint32(valueOffset + 4, isLittleEndian);
        return den === 0 ? 0 : num / den;
      }
      return null;
    }

    // APP1 마커(0xFFE1)에서 EXIF 탐색
    while (offset + 4 <= dv.byteLength) {
      if (dv.getUint8(offset) !== 0xFF) break;
      const marker = dv.getUint8(offset + 1);
      const segLen = dv.getUint16(offset + 2, false);
      if (marker === 0xE1 && offset + 10 <= dv.byteLength) {
        if (dv.getUint8(offset+4)===0x45 && dv.getUint8(offset+5)===0x78 &&
            dv.getUint8(offset+6)===0x69 && dv.getUint8(offset+7)===0x66) {
          tiffStart = offset + 10;
          break;
        }
      }
      offset += 2 + segLen;
    }
    if (tiffStart < 0) return null;

    const byteOrder = dv.getUint16(tiffStart, false);
    isLittleEndian = (byteOrder === 0x4949);

    const ifd0Offset = dv.getUint32(tiffStart + 4, isLittleEndian);
    const ifd0Abs = tiffStart + ifd0Offset;
    const ifd0Count = dv.getUint16(ifd0Abs, isLittleEndian);

    // IFD0에서 Model 태그(0x0110) 탐색
    for (let i = 0; i < ifd0Count; i++) {
      const e = ifd0Abs + 2 + i * 12;
      const tag = dv.getUint16(e, isLittleEndian);
      if (tag === 0x0110) { // Model tag
        const format = dv.getUint16(e + 2, isLittleEndian);
        const components = dv.getUint32(e + 4, isLittleEndian);
        if (format === 2) {
          model = readEntryValue(e, format, components);
          const modelLower = (model || '').toLowerCase();
          if (modelLower.includes('iphone')) cameraKey = 'iphone';
          else if (modelLower.includes('galaxy') || modelLower.includes('sm-')) cameraKey = 'galaxy';
          else if (modelLower.includes('canon') || modelLower.includes('nikon') || modelLower.includes('sony') || modelLower.includes('pentax')) cameraKey = 'dslr';
          else if (modelLower.includes('gopro') || modelLower.includes('dji') || modelLower.includes('action')) cameraKey = 'action';
          else if (modelLower.includes('cctv') || modelLower.includes('hikvision')) cameraKey = 'cctv';
        }
      } else if (tag === 0x8769) { // ExifIFDPointer
        exifIfdOffset = dv.getUint32(e + 8, isLittleEndian);
      }
    }

    if (exifIfdOffset >= 0) {
      const exifAbs = tiffStart + exifIfdOffset;
      const exifCount = dv.getUint16(exifAbs, isLittleEndian);
      for (let i = 0; i < exifCount; i++) {
        const e = exifAbs + 2 + i * 12;
        const tag = dv.getUint16(e, isLittleEndian);
        const format = dv.getUint16(e + 2, isLittleEndian);
        const components = dv.getUint32(e + 4, isLittleEndian);
        if (tag === 0x8827) { // ISO SpeedRatings
          const value = readEntryValue(e, format, components);
          if (Number.isFinite(Number(value))) iso = Number(value);
        } else if (tag === 0x829A) { // ExposureTime
          const value = readEntryValue(e, format, components);
          if (Number.isFinite(Number(value))) exposureMs = Math.max(1, Math.round(Number(value) * 1000));
        }
      }
    }

    return {
      model,
      cameraKey,
      iso,
      exposureMs,
    };
  } catch (e) {
    console.warn('카메라 모델 파싱 오류:', e);
    return null;
  }
}

/**
 * JPEG ArrayBuffer에서 EXIF GPS 좌표를 파싱합니다.
 * canvas 압축 후 EXIF가 제거되기 때문에, 압축 전 원본 버퍼에서 미리 추출합니다.
 * @returns {{lat: number, lon: number}|null}
 */
function extractGpsFromArrayBuffer(buffer) {
  try {
    const dv = new DataView(buffer);
    // JPEG 시그니처 확인
    if (dv.getUint16(0, false) !== 0xFFD8) return null;

    let offset = 2;
    let tiffStart = -1;
    let isLittleEndian = false;

    // APP1 마커(0xFFE1)에서 EXIF 탐색
    while (offset + 4 <= dv.byteLength) {
      if (dv.getUint8(offset) !== 0xFF) break;
      const marker = dv.getUint8(offset + 1);
      const segLen = dv.getUint16(offset + 2, false); // 빅 엔디안으로 먼저 읽기
      if (marker === 0xE1 && offset + 10 <= dv.byteLength) {
        // "Exif\0\0" 확인
        if (dv.getUint8(offset+4)===0x45 && dv.getUint8(offset+5)===0x78 &&
            dv.getUint8(offset+6)===0x69 && dv.getUint8(offset+7)===0x66) {
          tiffStart = offset + 10;
          break;
        }
      }
      offset += 2 + segLen;
    }
    if (tiffStart < 0) return null;

    // TIFF 헤더에서 바이트 오더 확인
    const byteOrder = dv.getUint16(tiffStart, false);
    isLittleEndian = (byteOrder === 0x4949); // 'II' = little endian, 'MM' = big endian

    // IFD0 오프셋
    const ifd0Offset = dv.getUint32(tiffStart + 4, isLittleEndian);
    const ifd0Abs = tiffStart + ifd0Offset;
    const ifd0Count = dv.getUint16(ifd0Abs, isLittleEndian);

    // IFD0에서 GPS IFD 오프셋(tag 0x8825) 탐색
    let gpsIfdOffset = -1;
    for (let i = 0; i < ifd0Count; i++) {
      const e = ifd0Abs + 2 + i * 12;
      const tag = dv.getUint16(e, isLittleEndian);
      if (tag === 0x8825) { // GPSInfo
        gpsIfdOffset = dv.getUint32(e + 8, isLittleEndian);
        break;
      }
    }
    if (gpsIfdOffset < 0) return null;

    const gpsAbs = tiffStart + gpsIfdOffset;
    const gpsCount = dv.getUint16(gpsAbs, isLittleEndian);

    function readRational(offset) {
      const num = dv.getUint32(offset, isLittleEndian);
      const den = dv.getUint32(offset + 4, isLittleEndian);
      return den === 0 ? 0 : num / den;
    }

    function readDegrees(offset) {
      const d = readRational(offset);
      const m = readRational(offset + 8);
      const s = readRational(offset + 16);
      return d + m / 60 + s / 3600;
    }

    let latRef = null, lonRef = null, lat = null, lon = null;
    for (let i = 0; i < gpsCount; i++) {
      const e = gpsAbs + 2 + i * 12;
      const tag = dv.getUint16(e, isLittleEndian);
      const format = dv.getUint16(e + 2, isLittleEndian);
      const components = dv.getUint32(e + 4, isLittleEndian);
      const valueOffset = dv.getUint32(e + 8, isLittleEndian);

      if (tag === 0x0001 && format === 2) { // GPSLatitudeRef
        latRef = String.fromCharCode(dv.getUint8(tiffStart + valueOffset));
      } else if (tag === 0x0002 && format === 5 && components === 3) { // GPSLatitude
        lat = readDegrees(tiffStart + valueOffset);
      } else if (tag === 0x0003 && format === 2) { // GPSLongitudeRef
        lonRef = String.fromCharCode(dv.getUint8(tiffStart + valueOffset));
      } else if (tag === 0x0004 && format === 5 && components === 3) { // GPSLongitude
        lon = readDegrees(tiffStart + valueOffset);
      }
    }

    if (lat === null || lon === null) return null;
    if (latRef === 'S') lat = -lat;
    if (lonRef === 'W') lon = -lon;
    return { lat, lon };
  } catch (e) {
    console.warn('GPS 파싱 오류:', e);
    return null;
  }
}

function compressImageToDataUrl(fileData, maxPx, quality) {
  // 이미지를 canvas로 먹심을 가진 쪽 maxPx 이하로 리사이즈 + JPEG 압축
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      let w = img.naturalWidth;
      let h = img.naturalHeight;
      if (w > maxPx || h > maxPx) {
        if (w >= h) { h = Math.round(h * maxPx / w); w = maxPx; }
        else        { w = Math.round(w * maxPx / h); h = maxPx; }
      }
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      resolve(canvas.toDataURL("image/jpeg", quality));
    };
    img.onerror = () => resolve(fileData); // 실패 시 원본 그대로
    img.src = fileData;
  });
}

function setSessionImage(fileName, fileData, fileSize) {
  sessionStorage.setItem("light_file", fileName);
  sessionStorage.setItem("light_size", fileSize);
  sessionStorage.setItem("light_time", new Date().toLocaleString());
  try {
    sessionStorage.setItem("light_image", fileData);
  } catch (e) {
    // QuotaExceededError: 이미 압축된 데이터라면 저장을 포기하고 계속 (이미압축 전용 코드에서 안전하게 수행)
    console.warn("sessionStorage 권한 초과, 이미지 저장 실패:", e);
  }
}

function readSessionImage() {
  const fallback = sampleImages[0]?.url || "";
  return {
    data: sessionStorage.getItem("light_image") || fallback,
    file: sessionStorage.getItem("light_file") || "샘플이미지.jpg",
    size: sessionStorage.getItem("light_size") || "743KB",
    time: sessionStorage.getItem("light_time") || new Date().toLocaleString()
  };
}

function resetExifSessionState() {
  [
    "light_rawGps",
    "light_exifCameraModel",
    "light_exifCameraKey",
    "light_gpsDetected",
    "light_allZonesMode",
    "light_zonesSummary",
    "light_zone",
    "light_zoneLabel",
  ].forEach((key) => sessionStorage.removeItem(key));
}

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0));
    else if (max === g) h = ((b - r) / d + 2);
    else h = ((r - g) / d + 4);
    h /= 6;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return { h, s, v };
}

function analyzeLightImage(imageSrc) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const w = Math.min(480, img.naturalWidth);
      const h = Math.min(360, img.naturalHeight);
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve({ objects: [], avgBrightness: 0, gamma: 1.8 });
        return;
      }
      ctx.drawImage(img, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      const data = imageData.data;

      let sumBrightness = 0;
      let sumV = 0;
      let sumV2 = 0;
      let pixels = 0;
      const clusters = [];

      function addCandidate(x, y, brightness, saturation) {
        const radius = 26;
        const found = clusters.find((c) => x >= c.minX - radius && x <= c.maxX + radius && y >= c.minY - radius && y <= c.maxY + radius);
        if (found) {
          found.minX = Math.min(found.minX, x);
          found.minY = Math.min(found.minY, y);
          found.maxX = Math.max(found.maxX, x);
          found.maxY = Math.max(found.maxY, y);
          found.sumBrightness += brightness;
          found.sumSaturation += saturation;
          found.count += 1;
          found.avgBrightness = found.sumBrightness / found.count;
          found.avgSaturation = found.sumSaturation / found.count;
        } else {
          clusters.push({ minX: x, minY: y, maxX: x, maxY: y, sumBrightness: brightness, sumSaturation: saturation, count: 1, avgBrightness: brightness, avgSaturation: saturation });
        }
      }

      for (let y = 0; y < h; y += 2) {
        for (let x = 0; x < w; x += 2) {
          const i = (y * w + x) * 4;
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          const hsv = rgbToHsv(r, g, b);
          const brightness = (r + g + b) / 3;
          sumBrightness += brightness;
          sumV += hsv.v;
          sumV2 += hsv.v * hsv.v;
          pixels += 1;

          const isStrongLight = hsv.v > 0.5 && (brightness > 150 || hsv.s > 0.25);
          const isModerateLight = hsv.v > 0.35 && brightness > 120;
          if (isStrongLight || isModerateLight) {
            addCandidate(x, y, brightness, hsv.s);
          }
        }
      }

      const avgBrightness = pixels ? Math.round(sumBrightness / pixels) : 0;
      const meanV = pixels ? sumV / pixels : 0;
      const varianceV = pixels ? sumV2 / pixels - meanV * meanV : 0;
      const gamma = Math.max(1.0, Math.min(3.0, 1.8 + varianceV * 2.5));

      const filtered = clusters.filter((c) => c.count >= 10 && c.avgBrightness > 120);
      const objects = filtered.slice(0, 5).map((c, idx) => {
        const width = c.maxX - c.minX;
        const height = c.maxY - c.minY;
        const area = width * height;
        const riskValue = Math.round((c.avgBrightness / 255) * 45 + c.avgSaturation * 35 + (gamma - 1.0) * 10);
        const riskLevel = riskValue >= 65 ? "위험" : riskValue >= 45 ? "주의" : "관찰";
        const label = idx === 0 ? "고휘도 간판" : idx === 1 ? "가로등" : "조명";
        return {
          name: `${label} ${idx + 1}`,
          type: idx === 0 ? "간판" : "조명",
          brightness: Math.round(c.avgBrightness),
          saturation: Number(c.avgSaturation.toFixed(2)),
          gamma: Number(gamma.toFixed(2)),
          riskValue,
          riskLevel,
          box: {
            x: Math.round((c.minX / w) * 100),
            y: Math.round((c.minY / h) * 100),
            width: Math.max(8, Math.round((width / w) * 100)),
            height: Math.max(8, Math.round((height / h) * 100))
          }
        };
      });

      const resultObjects = objects.length ? objects : [{ name: "야간 조명", type: "조명", brightness: avgBrightness, saturation: 0.3, gamma: Number(gamma.toFixed(2)), riskValue: 34, riskLevel: "관찰", box: { x: 26, y: 22, width: 25, height: 18 } }];
      resolve({ objects: resultObjects, avgBrightness, gamma });
    };
    img.onerror = () => {
      resolve({ objects: [{ name: "야간 조명", type: "조명", brightness: 35, saturation: 0.2, gamma: 1.9, riskValue: 35, riskLevel: "관찰", box: { x: 25, y: 20, width: 30, height: 16 } }], avgBrightness: 35, gamma: 1.9 });
    };
    img.src = imageSrc;
  });
}

function calculateRiskScore(detectedObjects, avgBrightness, gamma) {
  let score = 30;
  detectedObjects.forEach((item) => {
    const brightnessScore = item.brightness > 200 ? 22 : item.brightness > 170 ? 16 : item.brightness > 130 ? 10 : 4;
    const saturationScore = item.saturation > 0.75 ? 18 : item.saturation > 0.45 ? 10 : 4;
    const gammaScore = gamma > 2.2 ? 16 : gamma > 2.0 ? 10 : 4;
    score += brightnessScore + saturationScore + gammaScore;
    score += item.type === "간판" ? 6 : 3;
  });
  score += avgBrightness > 150 ? 10 : 0;
  score = Math.min(100, Math.max(0, score));
  let level = "관찰";
  if (score >= 80) level = "고위험";
  else if (score >= 60) level = "주의";
  return { score, level };
}

async function callApiAnalyze(imageData, captureSettings = {}) {
  try {
    const body = {
      image: imageData,
      cameraModel: captureSettings.cameraModel || sessionStorage.getItem("light_cameraModel") || "smartphone",
      iso: Number(captureSettings.iso ?? sessionStorage.getItem("light_cameraIso") ?? 100),
      exposureMs: Number(captureSettings.exposureMs ?? sessionStorage.getItem("light_cameraExposure") ?? 12),
      angleDeg: Number(captureSettings.angleDeg ?? sessionStorage.getItem("light_cameraAngle") ?? 0),
      cameraLabel: captureSettings.cameraLabel || sessionStorage.getItem("light_cameraLabel") || "스마트폰 기본",
      locationMode: captureSettings.locationMode || sessionStorage.getItem("light_locationMode") || "auto",
      locationZone: captureSettings.locationZone || sessionStorage.getItem("light_locationZone") || "제3종",
    };
    
    // canvas 압축으로 EXIF가 사라지므로, 업로드 시 미리 추출한 GPS를 같이 전송
    const rawGps = sessionStorage.getItem("light_rawGps");
    if (rawGps) {
      try {
        const gps = JSON.parse(rawGps);
        if (gps && typeof gps.lat === "number") {
          body.gpsLat = gps.lat;
          body.gpsLon = gps.lon;
        }
      } catch (e) { /* 파싱 실패 시 무시 */ }
    }
    
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (!response.ok) throw new Error(`API error ${response.status}`);
    const result = await response.json();
    if (result.status !== "success") throw new Error(result.message || "API 분석 실패");
    return result;
  } catch (err) {
    console.warn("API 분석 실패, 로컬 대체 실행", err);
    return simulateApiAnalysis(imageData);
  }
}

async function resolveImageDataForApi(imageRef) {
  if (!imageRef) return null;
  if (String(imageRef).startsWith("data:image")) return imageRef;
  // URL인 경우 fetch 후 base64로 변환
  try {
    const response = await fetch(imageRef);
    if (!response.ok) throw new Error(`image fetch failed: ${response.status}`);
    const blob = await response.blob();
    return await new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = reject;
      fr.readAsDataURL(blob);
    });
  } catch (err) {
    console.warn("이미지 URL을 data URL로 변환 실패 — 분석 불가", err);
    return null; // null 반환 → callApiAnalyze에서 바로 에러 처리
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function getStoredGps() {
  const value = sessionStorage.getItem("light_rawGps");
  if (!value) return null;
  try {
    const gps = JSON.parse(value);
    if (gps && Number.isFinite(Number(gps.lat)) && Number.isFinite(Number(gps.lon))) {
      return { lat: Number(gps.lat), lon: Number(gps.lon) };
    }
  } catch (e) {
    return null;
  }
  return null;
}

function formatDistanceMeters(value) {
  const distance = Number(value);
  if (!Number.isFinite(distance)) return "-";
  return distance < 10 ? `${distance.toFixed(2)}m` : `${distance.toFixed(1)}m`;
}

function renderCommercialGroup(title, stores) {
  const items = stores.map((store) => `
    <details class="commercial-store">
      <summary>
        <span>${escapeHtml(store.name || "상점명 없음")}</span>
        <small>${formatDistanceMeters(store.distanceMeters)}</small>
      </summary>
      <dl>
        <div><dt>상점명</dt><dd>${escapeHtml(store.name || "-")}</dd></div>
        <div><dt>영업상태</dt><dd>${escapeHtml(store.status || "-")}</dd></div>
        <div><dt>대분류</dt><dd>${escapeHtml(store.major || "-")}</dd></div>
        <div><dt>중분류</dt><dd>${escapeHtml(store.minor || "-")}</dd></div>
        <div><dt>업종명</dt><dd>${escapeHtml(store.businessType || "-")}</dd></div>
        <div><dt>도로명주소</dt><dd>${escapeHtml(store.address || "-")}</dd></div>
        <div><dt>개업일</dt><dd>${escapeHtml(store.openDate || "-")}</dd></div>
        <div><dt>폐업일</dt><dd>${escapeHtml(store.closeDate || "-")}</dd></div>
        <div><dt>거리</dt><dd>${formatDistanceMeters(store.distanceMeters)}</dd></div>
      </dl>
    </details>
  `).join("");

  return `
    <div class="commercial-group">
      <h4>${title}</h4>
      ${items || '<p class="commercial-empty">해당 반경 안에 포함되는 가게가 없습니다.</p>'}
    </div>
  `;
}

function setupBusanCommercialLookup(rawGps) {
  const lookupBtn = document.getElementById("commercialLookupBtn");
  const messageEl = document.getElementById("commercialGpsMessage");
  const resultsEl = document.getElementById("commercialResults");
  if (!lookupBtn || !messageEl || !resultsEl) return;

  const gps = rawGps && Number.isFinite(Number(rawGps.lat)) && Number.isFinite(Number(rawGps.lon))
    ? { lat: Number(rawGps.lat), lon: Number(rawGps.lon) }
    : null;

  if (!gps) {
    lookupBtn.style.display = "none";
    lookupBtn.disabled = true;
    messageEl.textContent = "이미지에 GPS 정보가 없어 주변 상점 조회를 사용할 수 없습니다.";
    resultsEl.innerHTML = "";
    return;
  }

  lookupBtn.style.display = "inline-flex";
  lookupBtn.disabled = false;
  messageEl.textContent = `GPS 좌표 ${gps.lat.toFixed(6)}, ${gps.lon.toFixed(6)} 기준으로 조회할 수 있습니다.`;

  lookupBtn.addEventListener("click", async () => {
    lookupBtn.disabled = true;
    lookupBtn.textContent = "조회 중...";
    resultsEl.innerHTML = '<p class="commercial-loading">주변 상점 정보를 불러오는 중입니다.</p>';

    try {
      const params = new URLSearchParams({
        lat: String(gps.lat),
        lon: String(gps.lon),
        address: "부산진구",
      });
      const response = await fetch(`/api/busan-commercial/nearby?${params.toString()}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== "success") {
        if (response.status === 409 && data.code === "CACHE_REQUIRED") {
          resultsEl.innerHTML = '<p class="commercial-empty">상점 데이터 캐시가 없습니다. 터미널에서 busan_update.py를 먼저 실행해주세요.</p>';
          return;
        }
        throw new Error(data.message || `상점 조회 실패 (${response.status})`);
      }

      const debug = data.debug || {};
      console.log("API Response:", debug.apiResponse ?? data);
      console.log("Cache Meta:", debug.cacheMeta);
      console.log("Items Count:", debug.itemsCount);
      console.log("Nearest stores TOP 10:", debug.nearestStores);
      console.log("Within 30m:", debug.within30mCount);

      const groups = data.groups || {};
      const within5m = groups.within5m || [];
      const within10m = groups.within10m || [];
      const within30m = groups.within30m || [];
      const nearestStores = data.nearestStores || [];
      const nearbyTotal = within5m.length + within10m.length + within30m.length;
      const nearbyEmptyMessage = nearbyTotal
        ? ""
        : '<p class="commercial-empty">30m 이내 주변 상점 정보가 없습니다.</p>';

      if (!nearbyTotal && !nearestStores.length) {
        resultsEl.innerHTML = '<p class="commercial-empty">30m 이내 주변 상점 정보가 없습니다.</p>';
        return;
      }

      resultsEl.innerHTML = [
        nearbyEmptyMessage,
        renderCommercialGroup("5m 이내", within5m),
        renderCommercialGroup("5m 초과 ~ 10m 이내", within10m),
        renderCommercialGroup("10m 초과 ~ 30m 이내", within30m),
        renderCommercialGroup("가장 가까운 상점 TOP 10", nearestStores),
      ].join("");
    } catch (err) {
      resultsEl.innerHTML = `<p class="commercial-error">${escapeHtml(err.message || "주변 상점 조회 중 오류가 발생했습니다.")}</p>`;
    } finally {
      lookupBtn.disabled = false;
      lookupBtn.textContent = "주변 상점 조회";
    }
  });
}

function simulateApiAnalysis(imageData) {
  return analyzeLightImage(imageData).then((analysis) => {
    // API 호출 실패 시 로컬 fallback — 과태료 산출 불가(서버 필요)하므로 미탐지로 반환
    return {
      status: "success",
      overall: "미탐지",
      totalFineAmount: 0,
      violationCount: 0,
      detected: [],
      riskSummary: "서버 연결 실패 — 과태료 판정 불가 (백엔드 서버를 실행해주세요)",
      zone: "제3종",
      zoneLabel: "주거지역",
      gpsDetected: false,
      avgBrightness: analysis.avgBrightness,
      model: "로컬 폴백 (서버 미연결)"
    };
  });
}

function mainPageInit() {
  const uploadInput = document.getElementById("uploadInput");
  const sampleBtn = document.getElementById("sampleBtn");

  if (uploadInput) {
    uploadInput.addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      resetExifSessionState();
      // 동일 파일 재선택 가능하도록 input 초기화
      uploadInput.value = "";

      function doCompress() {
        const reader = new FileReader();
        reader.onload = () => {
          // 대용량 사진은 압축 후 저장 (maxPx=1920, quality=0.85)
          compressImageToDataUrl(reader.result, 1920, 0.85).then((compressed) => {
            const capture = saveCaptureSettings();
            setSessionImage(file.name, compressed, `${Math.round(file.size / 1024)}KB`);
            sessionStorage.setItem("light_captureSummary", formatCaptureSummary(capture));
            
            // 미리보기 모달 표시
            showImagePreview(compressed, capture, file.name);
          });
        };
        reader.onerror = () => {
          alert("파일을 읽는 중 오류가 발생했습니다. 다른 이미지를 선택해주세요.");
        };
        reader.readAsDataURL(file);
      }

      // canvas 압축 전에 원본 ArrayBuffer에서 EXIF 정보(카메라 모델, GPS) 먼저 추출
      // (canvas.toDataURL() 은 EXIF를 제거하므로 반드시 원본 단계에서 추출해야 함)
      const exifReader = new FileReader();
      exifReader.onload = () => {
        const cameraInfo = extractCameraModelFromArrayBuffer(exifReader.result) || {};
        const gps = extractGpsFromArrayBuffer(exifReader.result);
        const cameraModel = cameraInfo.cameraKey || "smartphone";
        const iso = Number.isFinite(cameraInfo.iso) ? cameraInfo.iso : 100;
        const exposureMs = Number.isFinite(cameraInfo.exposureMs) ? cameraInfo.exposureMs : 12;
        const angleDeg = 0;
        const locationMode = "auto";
        const locationZone = "제3종";

        sessionStorage.setItem("light_exifCameraModel", cameraInfo.model || "기본 스마트폰");
        sessionStorage.setItem("light_exifCameraKey", cameraModel);
        sessionStorage.setItem("light_cameraModel", cameraModel);
        sessionStorage.setItem("light_cameraLabel", CAMERA_MODEL_LABELS[cameraModel] || CAMERA_MODEL_LABELS.default);
        sessionStorage.setItem("light_cameraIso", String(iso));
        sessionStorage.setItem("light_cameraExposure", String(exposureMs));
        sessionStorage.setItem("light_cameraAngle", String(angleDeg));
        sessionStorage.setItem("light_locationMode", locationMode);
        sessionStorage.setItem("light_locationZone", locationZone);

        const exifInfoEl = document.getElementById("exifDetectionInfo");
        const cameraTextEl = document.getElementById("detectedCameraModelText");
        const gpsTextEl = document.getElementById("detectedGpsText");
        const captureTextEl = document.getElementById("detectedCaptureText");
        const autoCameraModelEl = document.getElementById("autoCameraModel");
        const autoCameraIsoEl = document.getElementById("autoCameraIso");
        const autoCameraExposureEl = document.getElementById("autoCameraExposure");
        const autoCameraAngleEl = document.getElementById("autoCameraAngle");
        const autoLocationZoneEl = document.getElementById("autoLocationZone");
        const zoneLabelMap = {
          '제1종': '제1종 자연환경 보존지역',
          '제2종': '제2종 농림지역',
          '제3종': '제3종 주거지역',
          '제4종': '제4종 상업·공업지역',
        };

        if (exifInfoEl) exifInfoEl.style.display = "block";
        if (cameraTextEl) cameraTextEl.textContent = cameraInfo.model || "기본 스마트폰";
        if (gpsTextEl) gpsTextEl.textContent = gps && typeof gps.lat === "number" && typeof gps.lon === "number"
          ? `${gps.lat.toFixed(6)}, ${gps.lon.toFixed(6)}`
          : "감지되지 않음 - 기본 구역 적용";
        if (captureTextEl) captureTextEl.textContent = `카메라 기종 ${CAMERA_MODEL_LABELS[cameraModel] || CAMERA_MODEL_LABELS.default}, ISO ${iso}, 노출 ${exposureMs}ms, 각도 ${angleDeg}°로 자동 설정됩니다.`;
        if (autoCameraModelEl) autoCameraModelEl.textContent = CAMERA_MODEL_LABELS[cameraModel] || CAMERA_MODEL_LABELS.default;
        if (autoCameraIsoEl) autoCameraIsoEl.textContent = String(iso);
        if (autoCameraExposureEl) autoCameraExposureEl.textContent = `${exposureMs}ms`;
        if (autoCameraAngleEl) autoCameraAngleEl.textContent = `${angleDeg}°`;
        if (autoLocationZoneEl) autoLocationZoneEl.textContent = gps
          ? `GPS 자동감지 (${gps.lat.toFixed(6)}, ${gps.lon.toFixed(6)})`
          : "GPS 정보 없음 - 기본값 제3종 주거지역 적용";

        // canvas 압축으로 EXIF가 사라지므로, 업로드 시 미리 추출한 GPS를 같이 전송
        const rawGps = extractGpsFromArrayBuffer(exifReader.result);
        if (rawGps) {
          sessionStorage.setItem("light_rawGps", JSON.stringify(rawGps));
        } else {
          sessionStorage.removeItem("light_rawGps");
        }
        doCompress();
      };
      exifReader.onerror = () => {
        sessionStorage.removeItem("light_rawGps");
        doCompress();
      };
      exifReader.readAsArrayBuffer(file);
    });
  }

  if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
      resetExifSessionState();
      const capture = saveCaptureSettings();
      const sample = getRandomSampleImage();
      setSessionImage(sample.name, sample.url, "1.1MB");
      sessionStorage.setItem("light_captureSummary", formatCaptureSummary(capture));
      window.location.href = withVersion("/analysis");
    });
  }
}

/**
 * 이미지 미리보기 모달 표시
 */
function showImagePreview(compressedImageDataUrl, captureSettings, fileName) {
  const modal = document.getElementById("imagePreviewModal");
  const previewImg = document.getElementById("previewImageDisplay");
  const previewExifResult = document.getElementById("previewExifResult");
  const previewExifStatus = document.getElementById("previewExifStatus");
  const closeBtn = document.getElementById("closePreviewBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");

  if (!modal) return;

  // 이미지 표시
  if (previewImg) {
    previewImg.src = compressedImageDataUrl;
  }

  // EXIF 감지 상태 표시
  const cameraModel = sessionStorage.getItem("light_cameraModel") || "smartphone";
  const cameraLabel = sessionStorage.getItem("light_cameraLabel") || "기본 스마트폰";
  const iso = sessionStorage.getItem("light_cameraIso") || "-";
  const exposure = sessionStorage.getItem("light_cameraExposure") || "-";
  const angle = sessionStorage.getItem("light_cameraAngle") || "-";
  const gpsData = getStoredGps();
  const hasGps = !!gpsData;

  let exifText = `<strong>✅ EXIF 정보 감지됨</strong><br>`;
  exifText += `📷 카메라 기종: ${cameraLabel}<br>`;
  exifText += `⚙️ ISO: ${iso}<br>`;
  exifText += `⏱️ 노출 시간: ${exposure}ms<br>`;
  exifText += `📐 촬영 각도: ${angle}°`;

  if (!hasGps) {
    exifText += `<br>📍 위치: GPS 정보 없음 (기본값 적용)`;
  } else {
    exifText += `<br>📍 위치: GPS 감지 (${gpsData.lat.toFixed(4)}, ${gpsData.lon.toFixed(4)})`;
  }

  if (previewExifResult) {
    previewExifResult.innerHTML = exifText;
  }

  // 모달 표시
  modal.classList.remove("hidden");

  // 닫기 버튼
  if (closeBtn) {
    closeBtn.onclick = () => {
      modal.classList.add("hidden");
    };
  }

  // 분석 시작 버튼
  if (analyzeBtn) {
    analyzeBtn.onclick = () => {
      modal.classList.add("hidden");
      window.location.href = withVersion("/analysis");
    };
  }

  // ESC 키로 닫기
  const handleEscape = (e) => {
    if (e.key === "Escape") {
      modal.classList.add("hidden");
      document.removeEventListener("keydown", handleEscape);
    }
  };
  document.addEventListener("keydown", handleEscape);
}

function analysisPageInit() {
  syncFooterContact();
  const { data, file, size, time } = readSessionImage();
  ["light_detected", "light_overall", "light_totalFine", "light_violationCount", "light_riskSummary", "light_modelStatus", "light_zone", "light_zoneLabel", "light_gpsDetected", "light_allZonesMode", "light_zonesSummary", "light_pollutionOverall", "light_pollutionSummary"].forEach((k) => sessionStorage.removeItem(k));
  const imageEl = document.getElementById("analysisImage");
  const fileNameEl = document.getElementById("fileName");
  const uploadTimeEl = document.getElementById("uploadTime");
  const fileSizeEl = document.getElementById("fileSize");
  const stepList = document.getElementById("stepList");
  const progressFill = document.getElementById("progressFill");
  const progressPercent = document.getElementById("progressPercent");
  const statusText = document.getElementById("statusText");
  const capture = {
    cameraModel: sessionStorage.getItem("light_cameraModel") || "smartphone",
    cameraLabel: sessionStorage.getItem("light_cameraLabel") || "스마트폰 기본",
    iso: Number(sessionStorage.getItem("light_cameraIso") || "100"),
    exposureMs: Number(sessionStorage.getItem("light_cameraExposure") || "12"),
    angleDeg: Number(sessionStorage.getItem("light_cameraAngle") || "0"),
  };

  const analysisCameraModelEl = document.getElementById("analysisCameraModel");
  const analysisCameraIsoEl = document.getElementById("analysisCameraIso");
  const analysisCameraExposureEl = document.getElementById("analysisCameraExposure");
  const analysisCameraAngleEl = document.getElementById("analysisCameraAngle");
  if (analysisCameraModelEl) analysisCameraModelEl.textContent = capture.cameraLabel;
  if (analysisCameraIsoEl) analysisCameraIsoEl.textContent = `${capture.iso}`;
  if (analysisCameraExposureEl) analysisCameraExposureEl.textContent = `${capture.exposureMs}ms`;
  if (analysisCameraAngleEl) analysisCameraAngleEl.textContent = `${capture.angleDeg}°`;

  const captureSummaryEl = document.getElementById("analysisCaptureSummary");
  if (captureSummaryEl) captureSummaryEl.title = formatCaptureSummary(capture);

  if (imageEl) imageEl.src = data;
  if (fileNameEl) fileNameEl.textContent = file;
  if (uploadTimeEl) uploadTimeEl.textContent = time;
  if (fileSizeEl) fileSizeEl.textContent = size;

  const steps = ["이미지 불러오기", "간판/조명 객체 탐지", "밝기 강도 분석", "법규 기준 비교", "결과 생성"];

  let currentIndex = 0;
  function updateStepList(index) {
    if (!stepList || !progressFill || !progressPercent || !statusText) return;
    stepList.innerHTML = "";
    steps.forEach((title, i) => {
      const item = document.createElement("div");
      item.className = "step-item " + (i < index ? "step-done" : i === index ? "step-active" : "step-pending");
      item.innerHTML = `<span>${title}</span><span>${i < index ? "✔" : i === index ? "" : ""}</span>`;
      stepList.appendChild(item);
    });
    const rate = Math.round(((index + 1) / steps.length) * 100);
    progressFill.style.width = `${rate}%`;
    progressPercent.textContent = `${rate}%`;
    statusText.textContent = `${steps[index]} 중...`;
  }

  updateStepList(currentIndex);
  const interval = setInterval(() => {
    if (currentIndex >= steps.length - 1) {
      clearInterval(interval);
      if (statusText) statusText.textContent = "AI 분석이 완료되었습니다. 결과 페이지로 이동합니다.";
      resolveImageDataForApi(data).then((apiInput) => callApiAnalyze(apiInput, capture)).then((apiResult) => {
        const objectNames = apiResult.detected.map((o) => {
          const unit = o.unit || (o.type === "가로등" ? "lux" : "cd/m²");
          const measured = o.measuredValue ?? (unit === "lux" ? o.illuminanceLux : o.luminanceCdM2) ?? o.brightness;
          const displayName = o.type === "간판" && o.storeName ? o.storeName : o.name;
          return `${displayName}(${Math.round(measured)} ${unit}, 추정)`;
        }).join(", ");
        const detectedObjectsEl = document.getElementById("detectedObjects");
        const detectedRiskEl = document.getElementById("detectedRisk");
        if (detectedObjectsEl) detectedObjectsEl.textContent = objectNames || "없음";
        if (detectedRiskEl) detectedRiskEl.textContent = apiResult.overall || "-";
        sessionStorage.setItem("light_detected", JSON.stringify(apiResult.detected));
        sessionStorage.setItem("light_overall", apiResult.overall || "미탐지");
        sessionStorage.setItem("light_totalFine", apiResult.totalFineAmount ?? 0);
        sessionStorage.setItem("light_violationCount", apiResult.violationCount ?? 0);
        sessionStorage.setItem("light_riskSummary", apiResult.riskSummary);
        sessionStorage.setItem("light_modelStatus", apiResult.model || "모델 상태 없음");
        sessionStorage.setItem("light_zone", apiResult.zone || "제3종");
        sessionStorage.setItem("light_zoneLabel", apiResult.zoneLabel || "주거지역");
        sessionStorage.setItem("light_gpsDetected", apiResult.gpsDetected ? "true" : "false");
        sessionStorage.setItem("light_allZonesMode", apiResult.allZonesMode ? "true" : "false");
        sessionStorage.setItem("light_zonesSummary", JSON.stringify(apiResult.zonesSummary || {}));
        sessionStorage.setItem("light_pollutionOverall", apiResult.overallPollutionCategory || "미탐지");
        sessionStorage.setItem("light_pollutionSummary", JSON.stringify(apiResult.pollutionCategorySummary || {}));
        if (apiResult.captureContext) {
          sessionStorage.setItem("light_captureSummary", `${apiResult.captureContext.cameraLabel} · ISO ${apiResult.captureContext.iso} · ${apiResult.captureContext.exposureMs}ms · ${apiResult.captureContext.angleDeg}°`);
        }
      }).catch((err) => {
        console.error("분석 결과 저장 실패", err);
      }).finally(() => {
        setTimeout(() => { window.location.href = withVersion("/result"); }, 250);
      });
      return;
    }
    currentIndex += 1;
    updateStepList(currentIndex);
  }, 900);

  document.getElementById("reuploadBtn")?.addEventListener("click", () => { window.location.href = withVersion("/"); });
  document.getElementById("cancelBtn")?.addEventListener("click", () => { window.location.href = withVersion("/"); });
}

function resultPageInit() {
  syncFooterContact();
  const { data, file, size, time } = readSessionImage();
  const resultImage = document.getElementById("resultImage");
  const resultFileName = document.getElementById("resultFileName");
  const resultTime = document.getElementById("resultTime");
  const resultDetected = document.getElementById("resultDetected");
  const riskSummary = document.getElementById("riskSummary");
  const summaryOverall = document.getElementById("summaryOverall");
  const summaryBadge = document.getElementById("summaryBadge");
  const summaryViolation = document.getElementById("summaryViolation");
  const summaryConfidence = document.getElementById("summaryConfidence");
  const modelStatusEl = document.getElementById("modelStatus");

  if (resultImage) resultImage.src = data;
  if (resultFileName) resultFileName.textContent = file;
  if (resultTime) resultTime.textContent = time;

  const detected = JSON.parse(sessionStorage.getItem("light_detected") || "[]");
  const storeNames = [...new Set(detected
    .filter((item) => item.type === "간판")
    .map((item) => item.storeName?.trim())
    .filter(Boolean))];
  const signboardCount = detected.filter((item) => item.type === "간판" || item.name === "light_signboard" || item.name === "street sign").length;
  const modelStatus = sessionStorage.getItem("light_modelStatus") || "모델 정보 없음";
  if (modelStatusEl) modelStatusEl.textContent = modelStatus;
  const overall = sessionStorage.getItem("light_overall") || "미탐지";
  const totalFine = Number(sessionStorage.getItem("light_totalFine") || "0");
  const violationCount = Number(sessionStorage.getItem("light_violationCount") || "0");
  const riskSum = sessionStorage.getItem("light_riskSummary") || "-";
  const zone = sessionStorage.getItem("light_zone") || "제3종";
  const zoneLabel = sessionStorage.getItem("light_zoneLabel") || "주거지역";
  const gpsDetected = sessionStorage.getItem("light_gpsDetected") === "true";
  const allZonesMode = sessionStorage.getItem("light_allZonesMode") === "true";
  const zonesSummary = JSON.parse(sessionStorage.getItem("light_zonesSummary") || "{}");
  const pollutionOverall = sessionStorage.getItem("light_pollutionOverall") || "미탐지";
  const pollutionSummary = JSON.parse(sessionStorage.getItem("light_pollutionSummary") || "{}");
  const captureSummary = sessionStorage.getItem("light_captureSummary") || "-";
  const rawGps = getStoredGps();
  const gpsText = rawGps && typeof rawGps.lat === "number" && typeof rawGps.lon === "number"
    ? `${rawGps.lat.toFixed(6)}, ${rawGps.lon.toFixed(6)}`
    : (gpsDetected ? `${zone} (${zoneLabel})` : "미확인");

  function buildReportPayload() {
    return {
      fileName: file,
      fileSize: size,
      analysisTime: time,
      generatedAt: new Date().toLocaleString(),
      overall,
      totalFineAmount: totalFine,
      violationCount,
      riskSummary: riskSum,
      modelStatus,
      zone,
      zoneLabel,
      gpsDetected,
      allZonesMode,
      zonesSummary,
      pollutionOverall,
      pollutionSummary,
      detected,
      rawGps,
      gpsText,
      captureSummary,
      captureContext: {
        cameraModel: sessionStorage.getItem("light_cameraModel") || "smartphone",
        cameraLabel: sessionStorage.getItem("light_cameraLabel") || "스마트폰 기본",
        iso: Number(sessionStorage.getItem("light_cameraIso") || "100"),
        exposureMs: Number(sessionStorage.getItem("light_cameraExposure") || "12"),
        angleDeg: Number(sessionStorage.getItem("light_cameraAngle") || "0"),
        locationMode: sessionStorage.getItem("light_locationMode") || "auto",
        locationZone: sessionStorage.getItem("light_locationZone") || "제3종",
      },
    };
  }

  function buildTextReport(payload) {
    const pollutionCountsText = payload.pollutionSummary?.counts
      ? `침입광 ${payload.pollutionSummary.counts["침입광"] || 0}건, 눈부심 ${payload.pollutionSummary.counts["눈부심"] || 0}건, 산란광 ${payload.pollutionSummary.counts["산란광"] || 0}건, 군집된빛 ${payload.pollutionSummary.counts["군집된빛"] || 0}건`
      : "-";
    const detectedText = (payload.detected || []).map((item) => {
      const stageText = item.violationStage || "준수";
      const fineText = item.fineAmount > 0 ? `${item.fineAmount}만원` : "없음";
      const measuredValue = item.measuredValue ?? item.illuminanceLux ?? item.luminanceCdM2 ?? item.brightness ?? "-";
      const unit = item.unit || (item.type === "가로등" ? "lux" : "cd/m²");
      const storeText = item.type === "간판" && item.storeName ? ` / 상호명: ${item.storeName}` : "";
      const ocrText = item.type === "간판" && item.ocrText ? ` / OCR: ${item.ocrText}` : "";
      return `${item.name || "-"}${storeText}${ocrText} / ${item.type || "-"} / ${item.pollutionCategory || "미분류"} / ${measuredValue}${typeof measuredValue === "number" ? ` ${unit}` : ""} / ${stageText} / ${fineText}`;
    }).join("\n");
    return [
      "빛 공해 법규 위반 탐지 리포트",
      `파일: ${payload.fileName}`,
      `파일 크기: ${payload.fileSize}`,
      `분석 시간: ${payload.analysisTime}`,
      `생성 시간: ${payload.generatedAt}`,
      `종합 판정: ${payload.overall}`,
      `대표 분류: ${payload.pollutionOverall}`,
      `조명환경관리구역: ${payload.zone} (${payload.zoneLabel})`,
      `GPS: ${payload.gpsText}`,
      `촬영 조건: ${payload.captureSummary || "-"}`,
      `총 과태료: ${payload.totalFineAmount}만원`,
      `위반 건수: ${payload.violationCount}건`,
      `모델 상태: ${payload.modelStatus}`,
      `위험 요약: ${payload.riskSummary}`,
      `유형별 집계: ${pollutionCountsText}`,
      `탐지 객체:`,
      detectedText || "-",
      "",
      "법적 참고 기준: 인공조명에 의한 빛공해 방지법, 동 시행령 제8조, 동 시행규칙 별표",
      "주의: 본 문서는 이미지 기반 추정 분석 결과로, 공식 법적 증빙 시에는 원본 사진과 현장 측정 자료를 함께 제출하십시오.",
    ].join("\n");
  }

  function buildPdfFileName(originalFileName) {
    const baseName = String(originalFileName || "light_pollution_report").trim();
    const stem = baseName.replace(/\.[^.]+$/, "") || "light_pollution_report";
    const safeName = stem.replace(/[^\w\-가-힣]+/g, "_").replace(/^_+|_+$/g, "") || "light_pollution_report";
    return `${safeName}.pdf`;
  }

  async function downloadReport() {
    const payload = buildReportPayload();
    const pdfFileName = buildPdfFileName(payload.fileName);
    try {
      const response = await fetch("/api/report/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`PDF generation failed: ${response.status}`);
      }
      const blob = await response.blob();
      const link = document.createElement("a");
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      link.download = pdfFileName;
      link.click();
      URL.revokeObjectURL(objectUrl);
      return;
    } catch (err) {
      console.warn("PDF 리포트 생성 실패, txt로 대체합니다.", err);
      const link = document.createElement("a");
      const blob = new Blob([buildTextReport(payload)], { type: "text/plain;charset=utf-8" });
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      link.download = "light_pollution_report.txt";
      link.click();
      URL.revokeObjectURL(objectUrl);
    }
  }

  if (summaryOverall) summaryOverall.textContent = allZonesMode ? "종합 판정: GPS 미확인" : `종합 판정: ${overall}`;
  if (summaryBadge) {
    summaryBadge.textContent = overall === "미탐지" ? "미탐지" : totalFine > 0 ? `과태료 ${totalFine}만원` : "법규 준수";
    summaryBadge.className = "badge " + (totalFine > 0 ? "badge-danger" : "badge-safe");
  }
  if (summaryViolation) summaryViolation.textContent = totalFine > 0 ? `${totalFine}만원` : "없음";
  if (summaryConfidence) summaryConfidence.textContent = `${violationCount}건`;
  if (resultDetected) {
    resultDetected.textContent = detected.map((d) => `${d.type === "간판" && d.storeName ? d.storeName : d.name}(${d.pollutionCategory || "미분류"}/${d.violationStage || '준수'})`).join(", ") || "탐지된 객체 없음";
  }
  const summaryStoreNames = document.getElementById("summaryStoreNames");
  if (summaryStoreNames) {
    summaryStoreNames.textContent = storeNames.length
      ? storeNames.join(", ")
      : signboardCount > 0 ? "간판명 인식 실패" : "탐지된 간판 없음";
  }
  if (riskSummary) riskSummary.textContent = riskSum;
  const resultCaptureEl = document.getElementById("resultCapture");
  if (resultCaptureEl) resultCaptureEl.textContent = captureSummary;

  const summaryDetectedCount = document.getElementById("summaryDetectedCount");
  const summaryViolationCount = document.getElementById("summaryViolationCount");
  const summaryMaxBright = document.getElementById("summaryMaxBright");
  const summaryRiskLevel = document.getElementById("summaryRiskLevel");
  const detailDetectedCount = document.getElementById("detailDetectedCount");
  const detailBrightCount = document.getElementById("detailBrightCount");
  const detailTopObject = document.getElementById("detailTopObject");
  const detailRiskLevel = document.getElementById("detailRiskLevel");
  const detailConfidence = document.getElementById("detailConfidence");

  const violationItems = detected.filter((d) => d.compliance === "위반");
  const maxFineObject = violationItems.reduce((best, x) => ((x.fineAmount || 0) > ((best?.fineAmount) || 0) ? x : best), violationItems[0]);
  const maxStage = violationItems.find((d) => d.violationStage === "3단계") || violationItems.find((d) => d.violationStage === "2단계") || violationItems.find((d) => d.violationStage === "1단계");

  if (summaryDetectedCount) summaryDetectedCount.textContent = `${detected.length}개`;
  if (summaryViolationCount) summaryViolationCount.textContent = `${violationItems.length}개`;
  if (summaryMaxBright) summaryMaxBright.textContent = maxFineObject ? `${maxFineObject.name}` : "-";
  if (summaryRiskLevel) summaryRiskLevel.textContent = totalFine > 0 ? `${totalFine}만원` : "없음";
  if (detailDetectedCount) detailDetectedCount.textContent = `${detected.length}개`;
  if (detailBrightCount) detailBrightCount.textContent = `${violationItems.length}개`;
  if (detailTopObject) detailTopObject.textContent = maxFineObject ? `${maxFineObject.name}` : "-";
  if (detailRiskLevel) detailRiskLevel.textContent = totalFine > 0 ? `${totalFine}만원` : "없음";
  if (detailConfidence) detailConfidence.textContent = maxStage ? maxStage.violationStage : "없음";

  const zoneText = allZonesMode ? "GPS 미확인 — 구역별 시뮬레이션" : `${zone} (${zoneLabel})${gpsDetected ? " · GPS 자동판별" : ""}`;
  const resultZoneEl = document.getElementById("resultZone");
  const summaryZoneEl = document.getElementById("summaryZone");
  if (resultZoneEl) resultZoneEl.textContent = zoneText;
  if (summaryZoneEl) summaryZoneEl.textContent = zoneText;
  setupBusanCommercialLookup(rawGps);

  // GPS 없는 경우 4개 구역 전체 시뮬레이션 테이블
  const allZonesSection = document.getElementById("allZonesSection");
  if (allZonesSection) {
    if (allZonesMode && Object.keys(zonesSummary).length > 0) {
      const stageColor = { "3단계": "#dc2626", "2단계": "#f59e0b", "1단계": "#f97316", "준수": "#16a34a" };
      const rows = ["제1종", "제2종", "제3종", "제4종"].map((zc) => {
        const s = zonesSummary[zc] || {};
        const color = stageColor[s.overall] || "#666";
        return `<tr>
          <td style="padding:8px 10px;border:1px solid #e5e7ef;">${zc}<br/><small style="color:#999;font-weight:normal;">${s.zoneLabel || ""}</small></td>
          <td style="padding:8px 10px;border:1px solid #e5e7ef;color:${color};font-weight:600;">${s.overall || "-"}</td>
          <td style="padding:8px 10px;border:1px solid #e5e7ef;">${s.violationCount || 0}건</td>
          <td style="padding:8px 10px;border:1px solid #e5e7ef;font-weight:600;">${(s.totalFineAmount || 0) > 0 ? s.totalFineAmount + "만원" : "없음"}</td>
        </tr>`;
      }).join("");
      allZonesSection.innerHTML = `
        <div class="card-head"><h3>📍 GPS 미확인 — 구역별 과태료 시뮬레이션</h3></div>
        <p style="font-size:.85rem;color:#666;margin:0 0 12px;">EXIF GPS 정보가 없어 구역을 특정할 수 없습니다. 촬영 위치가 각 구역일 경우의 예상 과태료입니다.</p>
        <table style="width:100%;border-collapse:collapse;font-size:.85rem;">
          <tr style="background:#f9f9f9;">
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7ef;">구역</th>
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7ef;">위반 단계</th>
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7ef;">위반 건수</th>
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7ef;">총 과태료</th>
          </tr>
          ${rows}
        </table>`;
      allZonesSection.style.display = "block";
    } else {
      allZonesSection.style.display = "none";
    }
  }

  const overlayContainer = document.getElementById("overlayContainer");
  // 팝업을 담는 별도 레이어
  const imageWrap = overlayContainer ? overlayContainer.closest(".analysis-image-wrap") : null;
  let popupLayer = imageWrap ? imageWrap.querySelector("#popupLayer") : null;
  if (imageWrap && !popupLayer) {
    popupLayer = document.createElement("div");
    popupLayer.id = "popupLayer";
    imageWrap.appendChild(popupLayer);
  }

  if (overlayContainer) {
    overlayContainer.innerHTML = "";
    if (popupLayer) popupLayer.innerHTML = "";

    // 팝업 ID 카운터
    let popupIdCounter = 0;

    // 횟에 클릭한 박스 모음
    const clickedSet = new Set();

    function removePopup(id) {
      const el = popupLayer.querySelector(`[data-popup-id="${id}"]`);
      if (el) el.remove();
      clickedSet.delete(id);
    }

    function showPopup(item, box, popupClass, measured, lawUnit) {
      // 설명창은 한 번에 하나만 표시한다.
      popupLayer.replaceChildren();
      const id = ++popupIdCounter;
      const popup = document.createElement("div");
      popup.className = `overlay-popup ${popupClass}`;
      popup.dataset.popupId = id;
      popup.style.pointerEvents = "all";

      const fineText = item.fineAmount > 0 ? `<span style="color:#d63939;font-weight:700;">${item.fineAmount}만원</span>` : '<span style="color:#1f9d5d;">없음</span>';
      const isSignboard = item.type === "간판";
      const storeInfo = isSignboard && item.storeName
        ? `상호명: <b>${escapeHtml(item.storeName)}</b><br/>`
        : "";
      const ocrInfo = isSignboard && item.ocrText
        ? `인식 글자: <span title="${escapeHtml(item.ocrText)}">${escapeHtml(item.ocrText)}</span> (${Math.round(Number(item.ocrConfidence || 0) * 100)}%)<br/>`
        : "";
      popup.innerHTML = `
        <button class="popup-close" title="닫기">×</button>
        <strong>${escapeHtml(isSignboard && item.storeName ? item.storeName : item.name)} (${escapeHtml(item.lightType || item.type)})</strong>
        ${storeInfo}${ocrInfo}
        빛 공해 분류: <b>${item.pollutionCategory || "미분류"}</b> <small>${item.pollutionCategoryDesc || ""}</small><br/>
        측정값(참고용 추정): <b>${Math.round(measured)} ${lawUnit}</b><br/>
        ROI 밝기 평균/95%: ${Math.round(item.brightness ?? 0)} / ${Math.round(item.brightnessP95 ?? item.brightness ?? 0)}<br/>
        밝은 픽셀 비율: ${Number(item.brightPixelRatio ?? 0).toFixed(1)}%<br/>
        기준치: ${item.threshold ?? "-"} ${lawUnit} <small>(${item.basis || "-"})</small><br/>
        준수 여부: <b>${item.compliance}</b>${item.violationStage ? " · " + item.violationStage : ""}<br/>
        과태료: ${fineText}
      `;

      popup.querySelector(".popup-close").addEventListener("click", (e) => {
        e.stopPropagation();
        removePopup(id);
      });
      popupLayer.appendChild(popup);

      // 실제 팝업 크기를 측정한 뒤 이미지 경계 안으로 위치를 보정한다.
      const padding = 8;
      const gap = 6;
      const layerWidth = popupLayer.clientWidth;
      const layerHeight = popupLayer.clientHeight;
      const boxLeft = box.offsetLeft;
      const boxTop = box.offsetTop;
      const boxRight = boxLeft + box.offsetWidth;
      const boxBottom = boxTop + box.offsetHeight;

      popup.style.maxWidth = `${Math.max(1, layerWidth - padding * 2)}px`;
      popup.style.maxHeight = `${Math.max(1, layerHeight - padding * 2)}px`;

      const popupWidth = popup.offsetWidth;
      const popupHeight = popup.offsetHeight;
      let left = boxLeft;
      let top;

      // 기본적으로 박스 아래에 표시하고, 공간이 부족하면 위로 전환한다.
      if (boxBottom + gap + popupHeight <= layerHeight - padding) {
        top = boxBottom + gap;
      } else if (boxTop - gap - popupHeight >= padding) {
        top = boxTop - gap - popupHeight;
      } else {
        top = Math.min(Math.max(boxTop, padding), layerHeight - padding - popupHeight);
      }

      // 오른쪽이 잘리면 박스 오른쪽에 맞춘 후 다시 이미지 안으로 제한한다.
      if (left + popupWidth > layerWidth - padding) left = boxRight - popupWidth;
      left = Math.min(Math.max(left, padding), layerWidth - padding - popupWidth);
      top = Math.min(Math.max(top, padding), layerHeight - padding - popupHeight);

      popup.style.left = `${left}px`;
      popup.style.top = `${top}px`;
      return id;
    }

    detected.forEach((item) => {
      const box = document.createElement("div");
      box.className = "overlay " + (item.violationStage === "3단계" ? "box-high" : item.violationStage ? "box-medium" : "box-safe");
      const popupClass = item.violationStage === "3단계" ? "popup-high" : item.violationStage ? "popup-medium" : "popup-safe";
      box.style.top = `${item.box.y}%`;
      box.style.left = `${item.box.x}%`;
      box.style.width = `${item.box.width}%`;
      box.style.height = `${item.box.height}%`;

      const lawUnit = item.unit || (item.type === "가로등" ? "lux" : "cd/m²");
      const measured = item.measuredValue ?? (lawUnit === "lux" ? item.illuminanceLux : item.luminanceCdM2) ?? item.brightness;

      let activePopupId = null;
      box.addEventListener("click", (e) => {
        e.stopPropagation();
        // 이미 열려 있으면 닫기
        if (activePopupId !== null && popupLayer.querySelector(`[data-popup-id="${activePopupId}"]`)) {
          removePopup(activePopupId);
          activePopupId = null;
        } else {
          activePopupId = showPopup(item, box, popupClass, measured, lawUnit);
        }
      });

      overlayContainer.appendChild(box);
    });

    // 이미지 영역 외 클릭 시 모든 팝업 닫기
    imageWrap && imageWrap.addEventListener("click", () => {
      if (popupLayer) popupLayer.innerHTML = "";
    });
  }

  document.getElementById("redoBtn")?.addEventListener("click", () => { window.location.href = withVersion("/"); });
  document.getElementById("downloadBtn")?.addEventListener("click", downloadReport);
}

function indexPageInit() {
  const openPolicyBtn = document.getElementById("openPolicyBtn");
  const closePolicyBtn = document.getElementById("closePolicyBtn");
  const policyModal = document.getElementById("policyModal");
  openPolicyBtn?.addEventListener("click", () => policyModal?.classList.remove("hidden"));
  closePolicyBtn?.addEventListener("click", () => policyModal?.classList.add("hidden"));
}

window.addEventListener("DOMContentLoaded", () => {
  syncFooterContact();
  const path = window.location.pathname;
  if (path === "/" || path.endsWith("index.html") || path.endsWith("\\")) {
    mainPageInit();
    indexPageInit();
  } else if (path === "/analysis" || path.endsWith("analysis.html")) {
    analysisPageInit();
  } else if (path === "/result" || path.endsWith("result.html")) {
    resultPageInit();
  }
});

// Re-apply contact info when browser restores a page from bfcache.
window.addEventListener("pageshow", () => {
  syncFooterContact();
});
