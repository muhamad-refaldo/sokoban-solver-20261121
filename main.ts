// Sokoban solver v1 — push-optimal, room by room.
//
// Approach: a level is a 1-wide hallway plus 1-4 small rooms (3-4 boxes each).
// Boxes never leave their room (a box pushed into the hallway can never come
// back: the player cannot stand on the far side of the hallway to push it
// back), so the level's push-optimum is the sum of its rooms' optima. Each
// room is solved on its own with A* over pushes (admissible, consistent
// heuristic: the minimum-weight matching of boxes to goals over precomputed
// lone-box push distances), with three sound prunings: dead squares (a box
// that can never reach a goal), frozen 2x2 blocks, and frozen boxes
// (fixpoint: a box that can never be pushed in any direction again).
// The search is budgeted by work (expansions), never by the clock, so two
// runs answer byte for byte alike. If the optimal search exhausts its
// budget for a room, a weighted (greedy) search tries for any solution
// instead: a non-optimal solution still scores 500*P*/pushes, an unsolved
// room scores 0.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DEBUG = process.env.MAKEFUN_DEBUG === '1';

// Work budgets (expansions, never the clock).
const ASTAR_BUDGET = 800_000; // per room, optimal search
const GREEDY_BUDGET = 400_000; // per room, weighted fallback
const TOTAL_BUDGET = 4_000_000; // whole run, safety valve under the 60 s wall

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };
type Room = { boxes: number[]; goals: number[]; done: boolean };
type SearchOutcome = { pushes: number; moves: string } | null;

let usedExpansions = 0;

/** A level in XSB text. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * dist[gi * size + s]: the fewest pushes for a lone box on square s to reach
 * goals[gi], other boxes ignored. Reverse breadth-first search over the push
 * graph: a box on x came from y = x - d, pushed by a player standing on
 * x - 2d, so both y and x - 2d must be open floor.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array {
  const size = wall.length;
  const dist = new Int32Array(goals.length * size).fill(-1);
  const queue = new Int32Array(size);
  for (let gi = 0; gi < goals.length; gi++) {
    const g = goals[gi]!;
    const base = gi * size;
    let qh = 0;
    let qt = 0;
    queue[qt++] = g;
    dist[base + g] = 0;
    while (qh < qt) {
      const x = queue[qh++];
      const cur = dist[base + x]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const y = x - d;
        const behind = x - 2 * d;
        if (y < 0 || y >= size || behind < 0 || behind >= size) continue;
        if (wall[y] || wall[behind] || dist[base + y]! >= 0) continue;
        dist[base + y] = cur + 1;
        queue[qt++] = y;
      }
    }
  }
  return dist;
}

/**
 * The level's rooms: cut the player's row (the hallway) where a box there
 * reaches no goal, and flood-fill what is left. Each part holding boxes is a
 * room. Null when the level is no clean set of rooms (a box in the cut, a
 * goal outside every part, or a part with unequal boxes and goals).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const dist = pushDistances(w, wall, goals);
  const live = new Uint8Array(size);
  for (let s = 0; s < size; s++) {
    for (let gi = 0; gi < goals.length; gi++) {
      if (dist[gi * size + s]! >= 0) {
        live[s] = 1;
        break;
      }
    }
  }
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const comp = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  const stack: number[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || comp[s] >= 0) continue;
    const id = rooms.length;
    comp[s] = id;
    stack.length = 0;
    stack.push(s);
    while (stack.length > 0) {
      const p = stack.pop()!;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (!wall[q] && !cut[q] && comp[q] < 0) {
          comp[q] = id;
          stack.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (comp[b]! < 0) return null;
    rooms[comp[b]!]!.boxes.push(b);
  }
  for (const g of goals) {
    if (comp[g]! < 0) return null;
    rooms[comp[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The shortest walk from `from` to `to` around walls and boxes, lower LURD. */
function walkTo(w: number, wall: Uint8Array, boxes: number[], from: number, to: number): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const occ = new Uint8Array(size);
  for (const b of boxes) occ[b] = 1;
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = WALK[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (either case) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * A* over pushes for one (sub-)level. A state is the boxes' squares plus the
 * player's reachable region, named by its lowest square. `weight` 1 is the
 * optimal search; a larger weight is the greedy fallback (finds a solution
 * fast, not necessarily optimal). Returns null when the budget runs out or
 * the level has no solution.
 */
function solvePushes(sub: Level, weight: number, maxExp: number): SearchOutcome {
  const { w, wall, goals, boxes, player } = sub;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dist = pushDistances(w, wall, goals);
  const n = goals.length;
  const distMin = new Int32Array(size).fill(-1);
  for (let s = 0; s < size; s++) {
    let m = -1;
    for (let gi = 0; gi < n; gi++) {
      const d = dist[gi * size + s]!;
      if (d >= 0 && (m < 0 || d < m)) m = d;
    }
    distMin[s] = m;
  }
  if (boxes.some((b) => distMin[b]! < 0)) return null;
  if (boxes.every((b) => goalSet[b])) return { pushes: 0, moves: '' };

  // Heuristic: the minimum-weight matching of boxes to goals over the
  // lone-box push distances (admissible, and consistent: one push moves one
  // box one square, so the matching changes by at most 1). For more than 5
  // boxes the permutations are too many, so fall back to the sum of minima.
  const useHungarian = n <= 5;
  let perms: number[][] = [];
  if (useHungarian) {
    const gen = (arr: number[], k: number): void => {
      if (k === 1) {
        perms.push(arr.slice());
        return;
      }
      for (let i = 0; i < k; i++) {
        gen(arr, k - 1);
        const j = k % 2 === 0 ? i : 0;
        const tmp = arr[j]!;
        arr[j] = arr[k - 1]!;
        arr[k - 1] = tmp;
      }
    };
    gen(goals.map((_, i) => i), goals.length);
  }
  const heuristic = (bs: number[]): number => {
    if (useHungarian) {
      let best = Infinity;
      for (const p of perms) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[p[i]! * size + bs[i]!]!;
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (const b of bs) {
      const d = distMin[b]!;
      if (d < 0) return Infinity;
      s += d;
    }
    return s;
  };

  // A state's key: its region's lowest square and its boxes' squares.
  const numericKey = Math.pow(size, n + 1) < 2 ** 52;
  const keyOf = (bs: number[], low: number): number | string => {
    if (numericKey) {
      let k = low;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return `${low}:${bs.join(',')}`;
  };

  // Flood-fill scratch: two stamp arrays, so a child's region (seenB) never
  // clobbers the parent's region (seenA) while pushes are generated.
  const seenA = new Int32Array(size);
  const seenB = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stampA = 0;
  let stampB = 0;
  const regionLowA = (from: number): { low: number; stamp: number } => {
    stampA++;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    seenA[from] = stampA;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      const up = p - w;
      if (!wall[up] && !occupied[up] && seenA[up] !== stampA) {
        seenA[up] = stampA;
        stack[sp++] = up;
      }
      const dn = p + w;
      if (!wall[dn] && !occupied[dn] && seenA[dn] !== stampA) {
        seenA[dn] = stampA;
        stack[sp++] = dn;
      }
      const lf = p - 1;
      if (!wall[lf] && !occupied[lf] && seenA[lf] !== stampA) {
        seenA[lf] = stampA;
        stack[sp++] = lf;
      }
      const rt = p + 1;
      if (!wall[rt] && !occupied[rt] && seenA[rt] !== stampA) {
        seenA[rt] = stampA;
        stack[sp++] = rt;
      }
    }
    return { low, stamp: stampA };
  };
  const regionLowB = (from: number): number => {
    stampB++;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    seenB[from] = stampB;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      const up = p - w;
      if (!wall[up] && !occupied[up] && seenB[up] !== stampB) {
        seenB[up] = stampB;
        stack[sp++] = up;
      }
      const dn = p + w;
      if (!wall[dn] && !occupied[dn] && seenB[dn] !== stampB) {
        seenB[dn] = stampB;
        stack[sp++] = dn;
      }
      const lf = p - 1;
      if (!wall[lf] && !occupied[lf] && seenB[lf] !== stampB) {
        seenB[lf] = stampB;
        stack[sp++] = lf;
      }
      const rt = p + 1;
      if (!wall[rt] && !occupied[rt] && seenB[rt] !== stampB) {
        seenB[rt] = stampB;
        stack[sp++] = rt;
      }
    }
    return low;
  };

  // Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
  // and boxes with a box off its goal: no box in such a block can ever move.
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b2 = a + 1;
      const c = a + w;
      const d2 = c + 1;
      if (!((wall[a] || occupied[a]) && (wall[b2] || occupied[b2]) && (wall[c] || occupied[c]) && (wall[d2] || occupied[d2]))) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b2] && !goalSet[b2]) || (occupied[c] && !goalSet[c]) || (occupied[d2] && !goalSet[d2])) return true;
    }
    return false;
  };

  // Whether some box off its goal is frozen: it cannot be pushed in any
  // direction, where a direction counts as blocked only by a wall or a box
  // that is itself frozen (other boxes may move away). Fixpoint from nothing.
  const frozenOffGoal = (nb: number[]): boolean => {
    const n2 = nb.length;
    const frozen = new Uint8Array(n2);
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < n2; i++) {
        if (frozen[i]) continue;
        const b = nb[i]!;
        let allBlocked = true;
        for (let k = 0; k < 4; k++) {
          const d = DY[k]! * w + DX[k]!;
          const tgt = b + d;
          const beh = b - d;
          const tgtBlocked = wall[tgt] === 1 || (occupied[tgt] === 1 && frozen[nb.indexOf(tgt)] === 1);
          const behBlocked = wall[beh] === 1 || (occupied[beh] === 1 && frozen[nb.indexOf(beh)] === 1);
          if (!tgtBlocked && !behBlocked) {
            allBlocked = false;
            break;
          }
        }
        if (allBlocked) {
          frozen[i] = 1;
          changed = true;
        }
      }
    }
    for (let i = 0; i < n2; i++) if (frozen[i] && !goalSet[nb[i]!]) return true;
    return false;
  };

  // Binary heap on [f, h, insertion order, state], deterministic ties.
  const heap: [number, number, number, number][] = [];
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  // The states in order found: boxes, the player's square after the last
  // push, parent, the push's direction, pushes so far, region low, key.
  const stBoxes: number[][] = [boxes.slice()];
  const stPlayer: number[] = [player];
  const stParent: number[] = [-1];
  const stPushDir: number[] = [-1];
  const stG: number[] = [0];
  const stLow: number[] = [0];
  const stKey: (number | string)[] = [''];
  const best = new Map<number | string, number>();

  for (const b of boxes) occupied[b] = 1;
  const startRegion = regionLowA(player);
  for (const b of boxes) occupied[b] = 0;
  stLow[0] = startRegion.low;
  stKey[0] = keyOf(boxes, startRegion.low);
  const h0 = heuristic(boxes);
  if (h0 === Infinity) return null;
  best.set(stKey[0], 0);
  let seq = 0;
  heapPush([h0, h0, seq++, 0]);
  let expansions = 0;
  let goalState = -1;
  while (heap.length > 0) {
    const entry = heapPop();
    const s = entry[3];
    const bs = stBoxes[s]!;
    if (best.get(stKey[s]!) !== s) continue; // superseded by a cheaper route
    if (bs.every((b) => goalSet[b])) {
      goalState = s;
      break;
    }
    if (expansions >= maxExp || usedExpansions >= TOTAL_BUDGET) return null;
    expansions++;
    usedExpansions++;
    const p0 = stPlayer[s]!;
    for (const b of bs) occupied[b] = 1;
    const region = regionLowA(p0);
    const rstamp = region.stamp;
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        const behind = b - d;
        if (wall[t] || occupied[t] || distMin[t]! < 0) continue;
        if (seenA[behind] !== rstamp) continue; // the player cannot reach the push square
        const nb = bs.slice();
        nb[bi] = t;
        nb.sort((a, b2) => a - b2);
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(t) || frozenOffGoal(nb)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const childLow = regionLowB(b);
        const key = keyOf(nb, childLow);
        const g2 = stG[s]! + 1;
        const prev = best.get(key);
        if (prev !== undefined && stG[prev]! <= g2) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const h2 = heuristic(nb);
        if (h2 === Infinity) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const idx = stBoxes.length;
        stBoxes.push(nb);
        stPlayer.push(b);
        stParent.push(s);
        stPushDir.push(k);
        stG.push(g2);
        stLow.push(childLow);
        stKey.push(key);
        best.set(key, idx);
        heapPush([g2 + weight * h2, h2, seq++, idx]);
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (goalState < 0) return null;

  // The push path, then the moves: walk to each push square, then push.
  const path: number[] = [];
  for (let s = goalState; s > 0; s = stParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let playerPos = player;
  let curBoxes = boxes.slice();
  for (const s of path) {
    const parentBoxes = stBoxes[stParent[s]!]!;
    const nextBoxes = stBoxes[s]!;
    const from = parentBoxes.find((b) => !nextBoxes.includes(b))!;
    const k = stPushDir[s]!;
    const d = DY[k]! * w + DX[k]!;
    moves += walkTo(w, wall, curBoxes, playerPos, from - d);
    moves += PUSH[k]!;
    playerPos = from;
    curBoxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** One level, room by room; the empty answer when a room gives up. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const r = solvePushes(level, 1, ASTAR_BUDGET) ?? solvePushes(level, 5, GREEDY_BUDGET);
    return r ? r.moves : '';
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const r = solvePushes(sub, 1, ASTAR_BUDGET) ?? solvePushes(sub, 5, GREEDY_BUDGET);
    if (r === null) return '';
    moves += r.moves;
    player = endOf(level.w, player, r.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const level = parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  const before = usedExpansions;
  const moves = solveLevel(level);
  if (DEBUG) console.error(`${id}: ${moves.length} moves, ${usedExpansions - before} expansions`);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
if (DEBUG) console.error(`total expansions: ${usedExpansions}`);
