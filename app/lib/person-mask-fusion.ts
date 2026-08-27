function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function personGate(confidence: number) {
  // DeepLab's low-confidence person halo often contains nearby furniture or
  // props. Keep the confident human body and let temporal stabilization carry
  // uncertain hair/limb edges across an occasional weak frame.
  const normalized = clamp((confidence - 0.2) / 0.52, 0, 1);
  return normalized * normalized * (3 - 2 * normalized);
}

export function fuseSubjectAndPersonMasks(
  subject: Float32Array,
  subjectWidth: number,
  subjectHeight: number,
  person: Float32Array,
  personWidth: number,
  personHeight: number,
) {
  if (subject.length !== subjectWidth * subjectHeight) {
    throw new Error('指定主角遮罩尺寸不正確');
  }
  if (person.length !== personWidth * personHeight) {
    throw new Error('人物語意遮罩尺寸不正確');
  }

  const fused = new Float32Array(subject.length);
  for (let y = 0; y < subjectHeight; y += 1) {
    const personY = Math.min(
      personHeight - 1,
      Math.floor(((y + 0.5) / subjectHeight) * personHeight),
    );
    for (let x = 0; x < subjectWidth; x += 1) {
      const personX = Math.min(
        personWidth - 1,
        Math.floor(((x + 0.5) / subjectWidth) * personWidth),
      );
      const subjectIndex = y * subjectWidth + x;
      const semanticIndex = personY * personWidth + personX;
      fused[subjectIndex] = clamp(subject[subjectIndex], 0, 1)
        * personGate(person[semanticIndex]);
    }
  }
  return fused;
}
