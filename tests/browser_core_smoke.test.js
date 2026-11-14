'use strict';

const assert = require('node:assert/strict');
const ECG = require('../web/core.js');

const expectedClasses = ['N', 'S', 'V', 'F', 'Q'];
assert.deepEqual(ECG.CLS, expectedClasses, 'AAMI browser class order should remain stable');
assert.equal(ECG.FS, 360, 'Browser demo sampling frequency should be 360 Hz');
assert.equal(ECG.FN.length, 25, 'Browser demonstration feature schema should contain 25 features');

const model = ECG.train();
assert.ok(model.nTrain > 0, 'Synthetic training set should not be empty');
assert.ok(model.nTest > 0, 'Synthetic test set should not be empty');
assert.ok(model.rough.reduct.length > 0, 'QuickReduct should select at least one feature');
assert.ok(model.rough.reduct.every(i => Number.isInteger(i) && i >= 0 && i < ECG.FN.length));
assert.equal(model.mA.rules.length > 0, true, 'Fuzzy-only model should contain rules');
assert.equal(model.mB.rules.length > 0, true, 'Hybrid model should contain rules');
assert.ok(model.evA.acc >= 0 && model.evA.acc <= 1, 'Fuzzy-only accuracy must be a valid proportion');
assert.ok(model.evB.acc >= 0 && model.evB.acc <= 1, 'Hybrid accuracy must be a valid proportion');
assert.equal(model.evA.cm.length, expectedClasses.length, 'Fuzzy-only confusion matrix should cover all classes');
assert.equal(model.evB.cm.length, expectedClasses.length, 'Hybrid confusion matrix should cover all classes');

console.log('Browser core smoke test passed.');
console.log(JSON.stringify({
  trainBeats: model.nTrain,
  testBeats: model.nTest,
  featureCount: ECG.FN.length,
  reductSize: model.rough.reduct.length,
  fuzzyOnlyAccuracy: Number(model.evA.acc.toFixed(4)),
  hybridAccuracy: Number(model.evB.acc.toFixed(4)),
}, null, 2));
