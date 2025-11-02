#!/usr/bin/env node
// Lightweight local image-classification server using @xenova/transformers
// POST /classify with image bytes (image/jpeg or image/png). Returns JSON array [{label, score}].

import { createServer } from 'node:http'
import { pipeline as createStreamPipeline } from 'node:stream'
import { promisify } from 'node:util'
const streamPipeline = promisify(createStreamPipeline)

const PORT = Number(process.env.LOCAL_VISION_PORT || 8001)
const MODEL = process.env.LOCAL_VISION_MODEL || process.env.CIVIC_VISION_MODEL || 'google/vit-base-patch16-224'

console.log(`[local-vision] starting… model=${MODEL} port=${PORT}`)

// Lazy load transformers
const { pipeline } = await import('@xenova/transformers')
const classify = await pipeline('image-classification', MODEL)
console.log('[local-vision] pipeline ready')

createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && req.url === '/') {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ ok: true, model: MODEL }))
      return
    }
    if (req.method === 'POST' && req.url === '/classify') {
      const contentType = req.headers['content-type'] || 'application/octet-stream'
      const chunks = []
      await new Promise((resolve, reject) => {
        req.on('data', c => chunks.push(c))
        req.on('end', resolve)
        req.on('error', reject)
      })
      const buf = Buffer.concat(chunks)
      // Wrap as Blob so transformers can detect mime
      const blob = new Blob([buf], { type: contentType })
      const out = await classify(blob)
      // Normalize to array of {label, score}
      const labels = Array.isArray(out) ? out.map(r => ({ label: String(r.label), score: Number(r.score) })) : []
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify(labels))
      return
    }
    res.writeHead(404)
    res.end('Not found')
  } catch (e) {
    console.error('[local-vision] error', e)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: e?.message || 'error' }))
  }
}).listen(PORT, () => {
  console.log(`[local-vision] listening on http://127.0.0.1:${PORT}`)
})
