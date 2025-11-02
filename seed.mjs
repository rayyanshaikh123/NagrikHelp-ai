#!/usr/bin/env node
// Simple traffic seeding script: creates 5 users (if not existing) and ~200 issues.
// Usage:
//   API_BASE=http://localhost:8080 node scripts/seed.mjs
// Optionally set COUNT (default 200) and USERS (default 5)

const API_BASE = process.env.API_BASE || 'http://localhost:8080'
const TOTAL = parseInt(process.env.COUNT || '200', 10)
const USER_COUNT = parseInt(process.env.USERS || '5', 10)

const categories = ['POTHOLE','GARBAGE','STREETLIGHT','WATER','OTHER']

function rand(arr){return arr[Math.floor(Math.random()*arr.length)]}
function randomCoords(){
  // Rough India bounding box
  const lat = 8 + Math.random()*20 // 8 to 28
  const lon = 68 + Math.random()*20 // 68 to 88
  return `${lat.toFixed(6)},${lon.toFixed(6)}`
}
function randomSentence(words=6){
  const pool = ['civic','issue','report','urgent','local','road','waste','light','water','repair','pending','public','safety','community','update']
  return Array.from({length:words},()=>rand(pool)).join(' ')
}

async function registerOrLogin(email, password, name){
  // Try register
  const reg = await fetch(`${API_BASE}/api/auth/register`,{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,email,password})
  })
  if(reg.ok){
    const data = await reg.json(); return data.token
  }
  // fallback login
  const login = await fetch(`${API_BASE}/api/auth/login`,{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password})
  })
  if(!login.ok) throw new Error(`Auth failed for ${email}: ${login.status}`)
  const data = await login.json(); return data.token
}

async function createIssue(token, idx){
  const title = `Auto Issue #${idx} ${randomSentence(3)}`
  const description = `${randomSentence(18)}.`
  const location = randomCoords()
  const category = rand(categories)
  const payload = { title, description, location, category }
  const res = await fetch(`${API_BASE}/api/issues`,{
    method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${token}`},body:JSON.stringify(payload)
  })
  if(!res.ok){
    throw new Error(`Create failed (${res.status})`)
  }
  return res.json()
}

// Added optional post-processing: random status transitions, votes, comments.
const DO_STATUS = process.env.DO_STATUS !== '0'
const DO_VOTES = process.env.DO_VOTES !== '0'
const DO_COMMENTS = process.env.DO_COMMENTS !== '0'

async function adminLogin(){
  const res = await fetch(`${API_BASE}/api/auth/login`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:'superadmin@nagrikhelp.com',password:'rayyan7'})})
  if(!res.ok){ console.warn('Admin login failed; status changes skipped'); return null }
  const data = await res.json(); return data.token
}

async function updateStatus(adminToken, issueId, target){
  if(!adminToken) return
  await fetch(`${API_BASE}/api/admin/issues/${issueId}/status`,{method:'PUT',headers:{'Content-Type':'application/json','Authorization':`Bearer ${adminToken}`},body:JSON.stringify({status:target})})
}
async function vote(token, issueId, value){
  await fetch(`${API_BASE}/api/issues/${issueId}/votes`,{method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${token}`},body:JSON.stringify({value})})
}
async function comment(token, issueId, text){
  await fetch(`${API_BASE}/api/issues/${issueId}/comments`,{method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${token}`},body:JSON.stringify({text})})
}

;(async()=>{
  console.log(`Seeding ~${TOTAL} issues across ${USER_COUNT} users -> ${API_BASE}`)
  const users = []
  for(let i=0;i<USER_COUNT;i++){
    const email = `seed_user_${i+1}@example.com`
    const password = 'SeedPass123!'
    const name = `Seed User ${i+1}`
    try {
      const token = await registerOrLogin(email,password,name)
      users.push({email,token})
      console.log(`User ready: ${email}`)
    } catch(e){
      console.error(`User auth error for ${email}`, e.message)
      process.exit(1)
    }
  }
  let created = 0
  const perUser = Math.ceil(TOTAL / users.length)
  for (const u of users){
    for (let i=0;i<perUser && created < TOTAL;i++){
      try {
        await createIssue(u.token, created+1)
        created++
        if(created % 25 === 0) console.log(`Created ${created}`)
        // small jitter
        await new Promise(r=>setTimeout(r, 30 + Math.random()*70))
      } catch(e){
        console.warn(`Issue create failed: ${e.message}`)
      }
    }
  }
  console.log(`Done. Created ${created} issues.`)
  if(created === 0) process.exit(0)

  const adminToken = DO_STATUS ? await adminLogin() : null
  const issueIds = []
  // fetch all issues (admin endpoint for completeness)
  try {
    const res = await fetch(`${API_BASE}/api/admin/issues`,{headers: adminToken? {Authorization:`Bearer ${adminToken}`} : {}})
    if(res.ok){
      const all = await res.json();
      for(const it of all){ issueIds.push(it.id) }
    }
  } catch{}

  console.log(`Post-processing ${issueIds.length} issues (status=${DO_STATUS}, votes=${DO_VOTES}, comments=${DO_COMMENTS})`)

  // Status transitions: ~40% -> IN_PROGRESS, of those ~50% -> RESOLVED
  if(DO_STATUS && adminToken){
    for(const id of issueIds){
      if(Math.random()<0.40){
        await updateStatus(adminToken,id,'IN_PROGRESS')
        if(Math.random()<0.50){
          await updateStatus(adminToken,id,'RESOLVED')
        }
      }
    }
    console.log('Status transitions applied')
  }

  // Votes
  if(DO_VOTES){
    for(const id of issueIds){
      const voteCount = Math.floor(Math.random()*11) // 0-10
      for(let v=0; v<voteCount; v++){
        const u = rand(users)
        const val = Math.random()<0.7 ? 'UP' : 'DOWN'
        vote(u.token,id,val)
      }
    }
    console.log('Vote requests scheduled')
  }

  // Comments
  if(DO_COMMENTS){
    const commentSnippets = ['Needs attention','Please fix soon','Any update?','Critical issue','Temporary workaround applied','Seen this too','Forwarded to authorities','Resolved for now','Still broken','Escalated']
    for(const id of issueIds){
      const cCount = Math.random()<0.6 ? Math.floor(Math.random()*6) : 0 // 0-5 ~60% chance
      for(let c=0;c<cCount;c++){
        const u = rand(users)
        const text = rand(commentSnippets) + (Math.random()<0.4? ' '+rand(commentSnippets):'')
        comment(u.token,id,text)
      }
    }
    console.log('Comment requests scheduled')
  }

  console.log('Seeding complete.')
  process.exit(0)
})().catch(e=>{console.error(e);process.exit(1)})
