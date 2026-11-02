import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

class MockXmlHttpRequest {
  static instances: MockXmlHttpRequest[] = []
  method = ''
  url = ''
  status = 201
  responseText = ''
  headers: Record<string, string> = {}
  body: Document | XMLHttpRequestBodyInit | null = null
  upload: XMLHttpRequestUpload = { onprogress: null } as XMLHttpRequestUpload
  onload: ((this: XMLHttpRequest, ev: ProgressEvent) => unknown) | null = null
  onerror: ((this: XMLHttpRequest, ev: ProgressEvent) => unknown) | null = null

  constructor() { MockXmlHttpRequest.instances.push(this) }
  open(method: string, url: string) { this.method = method; this.url = url }
  setRequestHeader(name: string, value: string) { this.headers[name] = value }
  send(body: Document | XMLHttpRequestBodyInit | null) {
    this.body = body
    this.upload.onprogress?.call(
      this as unknown as XMLHttpRequest,
      new ProgressEvent('progress', { lengthComputable: true, loaded: 4, total: 4 }) as ProgressEvent<XMLHttpRequestEventTarget>,
    )
    this.onload?.call(this as unknown as XMLHttpRequest, new ProgressEvent('load'))
  }
}

beforeEach(() => {
  vi.resetModules()
  MockXmlHttpRequest.instances = []
  window.__FOOTBALLAI_CONFIG__ = { apiBase: '', uploadMode: 'direct' }
  vi.stubGlobal('XMLHttpRequest', MockXmlHttpRequest)
})

afterEach(() => {
  delete window.__FOOTBALLAI_CONFIG__
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('cloud direct upload', () => {
  it('authorizes, PUTs bytes to object storage, then finalizes metadata', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        run_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        upload: {
          method: 'PUT',
          url: 'https://example.blob.core.windows.net/container/object?redacted',
          headers: { 'x-ms-blob-type': 'BlockBlob', 'Content-Type': 'video/mp4' },
          max_bytes: 1024,
          required_content_type: 'video/mp4',
        },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ run_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa' }), {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      }))
    vi.stubGlobal('fetch', fetchMock)
    const { uploadAnalysis } = await import('./api')
    const form = new FormData()
    form.set('video', new File(['tiny'], 'tiny.mp4', { type: 'video/mp4' }))
    form.set('match_name', 'Azure demo')
    form.set('data_origin', 'evaluation')
    form.set('pipeline_profile', 'demo_fast')
    const progress = vi.fn()

    const result = await uploadAnalysis(form, progress)

    expect(result.run_id).toBe('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/uploads/authorize')
    expect(fetchMock.mock.calls[1][0]).toBe('/api/v1/uploads/finalize')
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
      match_name: 'Azure demo', data_origin: 'evaluation', pipeline_profile: 'demo_fast',
    })
    expect(MockXmlHttpRequest.instances).toHaveLength(1)
    expect(MockXmlHttpRequest.instances[0]).toMatchObject({
      method: 'PUT',
      headers: { 'x-ms-blob-type': 'BlockBlob', 'Content-Type': 'video/mp4' },
    })
    expect(MockXmlHttpRequest.instances[0].body).toBeInstanceOf(File)
    expect(progress).toHaveBeenLastCalledWith(100)
  })
})

describe('explainRun', () => {
  it('posts the question as JSON and sends no credential', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response('{}', { status: 200 })))
    vi.stubGlobal('fetch', fetchMock)
    const { explainRun } = await import('./api')
    await explainRun('run-1', 'How reliable was the tracking?')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/v1/runs/run-1/explain')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ question: 'How reliable was the tracking?' })
    expect(Object.keys(init.headers as Record<string, string>).map((name) => name.toLowerCase())).toEqual(['content-type', 'accept'])
  })

  it('asks the agent on its own route, with the run only in the path', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response('{}', { status: 200 })))
    vi.stubGlobal('fetch', fetchMock)
    const { askAgent } = await import('./api')
    await askAgent('run-1', 'What evidence is missing?')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/v1/runs/run-1/agent')
    expect(JSON.parse(String(init.body))).toEqual({ question: 'What evidence is missing?' })
  })

  it('raises the server message and status on failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify({ detail: { message: 'Only completed runs.' } }), { status: 409 }))))
    const { explainRun, ApiError } = await import('./api')
    const failure = await explainRun('run-1').catch((reason: unknown) => reason)
    expect(failure).toBeInstanceOf(ApiError)
    expect(failure).toMatchObject({ message: 'Only completed runs.', status: 409 })
  })
})
