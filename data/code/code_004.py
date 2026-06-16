async function encodeBatch(docs) {
  return Promise.all(docs.map(async (doc) => ({
    bytes: Buffer.byteLength(doc, 'utf8'),
    preview: doc.slice(0, 32),
  })));
}
