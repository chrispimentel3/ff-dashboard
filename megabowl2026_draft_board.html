<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mega Bowl 2026 — Draft Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;800&family=Archivo+Narrow:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:        #F2F1EE;
    --surface:   #FFFFFF;
    --surface2:  #ECEAE6;
    --border:    #D8D5CF;
    --border2:   #C8C4BC;
    --ink:       #1C1C1A;
    --ink2:      #5A574F;
    --ink3:      #9A9690;
    --mine-bg:   #FFFBEF;
    --mine-acc:  #D4960A;
    --mine-brd:  #E4B34A;
    --spot-bg:   #EEF4FF;

    --qb:#C2452D;
    --rb:#2E6DB4;
    --wr:#B87A10;
    --te:#6B3FA0;
    --k: #5A7080;
    --def:#3A7048;
    --qb-t:#fff;--rb-t:#fff;--wr-t:#fff;--te-t:#fff;--k-t:#fff;--def-t:#fff;
  }

  *{box-sizing:border-box;margin:0;padding:0;}
  html{font-size:13px;}
  body{
    background:var(--bg);
    color:var(--ink);
    font-family:'Archivo Narrow',system-ui,sans-serif;
    -webkit-font-smoothing:antialiased;
    height:100vh;
    display:flex;
    flex-direction:column;
    overflow:hidden;
  }

  /* ── masthead ── */
  header{
    flex:0 0 auto;
    background:var(--surface);
    border-bottom:1px solid var(--border);
    padding:14px 20px 12px;
    display:flex;
    align-items:baseline;
    gap:18px;
    flex-wrap:wrap;
  }
  .title{
    font-family:'Archivo',sans-serif;
    font-weight:800;
    font-size:22px;
    letter-spacing:-0.02em;
    color:var(--ink);
    white-space:nowrap;
  }
  .title em{font-style:normal;color:var(--mine-acc);}
  .subtitle{font-size:12px;color:var(--ink3);line-height:1.4;}

  /* ── controls ── */
  .controls{
    flex:0 0 auto;
    background:var(--surface2);
    border-bottom:1px solid var(--border);
    padding:8px 20px;
    display:flex;
    align-items:center;
    gap:16px;
    flex-wrap:wrap;
  }
  .chipset{display:flex;gap:4px;align-items:center;}
  .clabel{font-size:11px;color:var(--ink3);margin-right:2px;}
  .chip{
    font-family:'Archivo Narrow',sans-serif;
    font-weight:700;font-size:11.5px;
    padding:3px 9px;border-radius:3px;
    border:1px solid var(--border2);
    background:var(--surface);color:var(--ink2);
    cursor:pointer;transition:background .1s,color .1s,border-color .1s;
  }
  .chip:hover{background:var(--border);}
  .chip[aria-pressed="true"]{border-color:transparent;}
  .chip[data-pos="ALL"][aria-pressed="true"]{background:var(--ink);color:#fff;}
  .chip[data-pos="QB"][aria-pressed="true"]{background:var(--qb);color:#fff;}
  .chip[data-pos="RB"][aria-pressed="true"]{background:var(--rb);color:#fff;}
  .chip[data-pos="WR"][aria-pressed="true"]{background:var(--wr);color:#fff;}
  .chip[data-pos="TE"][aria-pressed="true"]{background:var(--te);color:#fff;}
  .chip[data-pos="K"][aria-pressed="true"]{background:var(--k);color:#fff;}
  .chip[data-pos="DEF"][aria-pressed="true"]{background:var(--def);color:#fff;}

  .search{
    font-family:'Archivo Narrow',sans-serif;font-size:12px;
    padding:4px 9px;border-radius:3px;
    border:1px solid var(--border2);background:var(--surface);
    color:var(--ink);width:160px;
  }
  .search::placeholder{color:var(--ink3);}
  .search:focus{outline:2px solid var(--mine-brd);outline-offset:1px;}
  .hint{font-size:11px;color:var(--ink3);margin-left:auto;}

  /* ── board wrapper ── */
  .board-wrap{
    flex:1 1 0;
    overflow:auto;
    position:relative;
  }

  /* ── the grid ── */
  /* 
     We use a real <table> so sticky first-col + sticky header both work 
     without fighting a CSS grid's min-width.
     Table fills viewport width exactly; cells share space equally.
  */
  table{
    width:100%;
    border-collapse:collapse;
    table-layout:fixed;
  }

  /* col widths */
  col.rail{width:38px;}
  /* team cols share remaining width equally via table-layout:fixed */

  /* ── header row ── */
  thead th{
    position:sticky;top:0;z-index:4;
    background:var(--surface);
    border-right:1px solid var(--border);
    border-bottom:2px solid var(--border2);
    padding:7px 6px 5px;
    vertical-align:top;
    cursor:pointer;user-select:none;
    text-align:left;
  }
  thead th.rail-th{
    cursor:default;
    background:var(--surface2);
    z-index:5;
    position:sticky;left:0;top:0;
  }
  thead th.mine-th{background:var(--mine-bg);border-bottom-color:var(--mine-brd);}
  thead th.spot-th{background:var(--spot-bg);}
  .th-seat{font-size:10px;color:var(--ink3);letter-spacing:0.05em;}
  .th-team{
    font-weight:700;font-size:13px;line-height:1.2;
    color:var(--ink);margin-top:2px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  thead th.mine-th .th-team{color:var(--mine-acc);}
  .th-tally{display:flex;gap:2px;margin-top:5px;flex-wrap:wrap;}
  .th-tally b{
    font-size:9.5px;font-weight:700;
    padding:1px 3px;border-radius:2px;color:#fff;
  }
  .th-tally b.QB{background:var(--qb);}
  .th-tally b.RB{background:var(--rb);}
  .th-tally b.WR{background:var(--wr);}
  .th-tally b.TE{background:var(--te);}
  .th-tally b.K{background:var(--k);}
  .th-tally b.DEF{background:var(--def);}

  /* ── round rail ── */
  td.rail{
    position:sticky;left:0;z-index:2;
    background:var(--surface2);
    border-right:1px solid var(--border2);
    border-bottom:1px solid var(--border);
    text-align:center;
    vertical-align:middle;
    padding:2px 0;
  }
  .rd-n{font-family:'Archivo',sans-serif;font-weight:800;font-size:14px;color:var(--ink3);line-height:1;}
  .rd-dir{font-size:10px;color:var(--border2);line-height:1;margin-top:1px;}

  /* ── pick cells ── */
  td.pick-cell{
    border-right:1px solid var(--border);
    border-bottom:1px solid var(--border);
    padding:3px;
    vertical-align:top;
  }
  td.pick-cell.mine-cell{background:var(--mine-bg);}
  td.pick-cell.spot-cell{background:var(--spot-bg);}

  .card{
    position:relative;
    background:var(--surface);
    border-left:4px solid #ccc;
    border-radius:2px;
    padding:3px 5px 4px 6px;
    height:100%;
    min-height:56px;
    transition:opacity .12s,filter .12s;
  }
  .card.QB{border-left-color:var(--qb);}
  .card.RB{border-left-color:var(--rb);}
  .card.WR{border-left-color:var(--wr);}
  .card.TE{border-left-color:var(--te);}
  .card.K {border-left-color:var(--k);}
  .card.DEF{border-left-color:var(--def);}
  .card.muted{opacity:0.15;filter:saturate(0.1);}
  .card.mine-card{box-shadow:0 0 0 2px var(--mine-brd);}

  .c-no{
    font-size:9.5px;font-weight:700;color:var(--ink3);
    position:absolute;top:3px;right:4px;
  }
  .c-name{
    font-weight:700;font-size:11.5px;line-height:1.2;
    color:var(--ink);padding-right:20px;
    margin-bottom:3px;
    /* clamp to two lines */
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
  }
  .c-meta{display:flex;align-items:center;gap:3px;}
  .c-pos{
    font-size:9.5px;font-weight:700;
    padding:1px 4px;border-radius:2px;color:#fff;flex:0 0 auto;
  }
  .c-pos.QB{background:var(--qb);}
  .c-pos.RB{background:var(--rb);}
  .c-pos.WR{background:var(--wr);}
  .c-pos.TE{background:var(--te);}
  .c-pos.K {background:var(--k);}
  .c-pos.DEF{background:var(--def);}
  .c-tm{font-size:10px;color:var(--ink3);font-weight:600;letter-spacing:0.04em;}

  /* footer */
  footer{
    flex:0 0 auto;
    background:var(--surface2);
    border-top:1px solid var(--border);
    padding:6px 20px;
    font-size:11px;color:var(--ink3);
  }

  @media (prefers-reduced-motion:reduce){*{transition:none !important;}}
</style>
</head>
<body>

<header>
  <h1 class="title">Mega Bowl 2026 &nbsp;<em>Draft Board</em></h1>
  <p class="subtitle">12 teams · 15 rounds · snake · half-PPR &nbsp;|&nbsp; <strong>TaylorMade = seat 8</strong> (gold). Click a team to spotlight their roster.</p>
</header>

<div class="controls">
  <div class="chipset">
    <span class="clabel">Filter</span>
    <button class="chip" data-pos="ALL" aria-pressed="true">All</button>
    <button class="chip" data-pos="QB"  aria-pressed="false">QB</button>
    <button class="chip" data-pos="RB"  aria-pressed="false">RB</button>
    <button class="chip" data-pos="WR"  aria-pressed="false">WR</button>
    <button class="chip" data-pos="TE"  aria-pressed="false">TE</button>
    <button class="chip" data-pos="K"   aria-pressed="false">K</button>
    <button class="chip" data-pos="DEF" aria-pressed="false">DEF</button>
  </div>
  <input class="search" id="search" type="search" placeholder="Find a player…" aria-label="Find a player">
  <span class="hint" id="hint">180 picks</span>
</div>

<div class="board-wrap">
  <table id="board">
    <colgroup>
      <col class="rail">
      <col><col><col><col><col><col><col><col><col><col><col><col>
    </colgroup>
    <thead><tr id="hrow"></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<footer>
  Odd rounds → left to right &nbsp;·&nbsp; Even rounds ← right to left &nbsp;·&nbsp; Corner number = overall pick &nbsp;·&nbsp; Colored left edge = position
</footer>

<script>
const TEAMS   = ["deez nuts","TyRick Hill","Undisputed","Any Given Ya…","L'Omar","I Chase Brow…","Bed Bath & B…","TaylorMade","JAH PLS","Play Action …","BillsMafia","Crabcakes an…"];
const MY_SEAT = 8;
const ORDER   = ["QB","RB","WR","TE","K","DEF"];

const ROUNDS = [
["Jahmyr Gibbs|RB|DET","Ja'Marr Chase|WR|CIN","Bijan Robinson|RB|ATL","Jonathan Taylor|RB|IND","Puka Nacua|WR|LAR","Saquon Barkley|RB|PHI","Amon-Ra St. Brown|WR|DET","Christian McCaffrey|RB|SF","James Cook III|RB|BUF","Jaxon Smith-Njigba|WR|SEA","Kenneth Walker III|RB|KC","CeeDee Lamb|WR|DAL"],
["Chase Brown|RB|CIN","Justin Jefferson|WR|MIN","Derrick Henry|RB|BAL","De'Von Achane|RB|MIA","Malik Nabers|WR|NYG","Omarion Hampton|RB|LAC","A.J. Brown|WR|NE","Lamar Jackson|QB|BAL","Nico Collins|WR|HOU","Kyren Williams|RB|LAR","Javonte Williams|RB|DAL","Drake London|WR|ATL"],
["George Pickens|WR|DAL","Travis Etienne Jr.|RB|NO","Brock Bowers|TE|LV","Ashton Jeanty|RB|LV","Zay Flowers|WR|BAL","Josh Allen|QB|BUF","Rashee Rice|WR|KC","Breece Hall|RB|NYJ","DeVonta Smith|WR|PHI","Jeremiyah Love|RB|ARI","Chris Olave|WR|NO","Trey McBride|TE|ARI"],
["Tee Higgins|WR|CIN","D'Andre Swift|RB|CHI","Ladd McConkey|WR|LAC","Jaylen Waddle|WR|DEN","Emeka Egbuka|WR|TB","Garrett Wilson|WR|NYJ","Cam Skattebo|RB|NYG","Bucky Irving|RB|TB","Tetairoa McMillan|WR|CAR","David Montgomery|RB|HOU","Colston Loveland|TE|CHI","Bhayshul Tuten|RB|JAX"],
["Luther Burden III|WR|CHI","Joe Burrow|QB|CIN","Terry McLaurin|WR|WAS","Sam LaPorta|TE|DET","Quinshon Judkins|RB|CLE","DJ Moore|WR|BUF","Tyler Warren|TE|IND","Jadarian Price|RB|SEA","Rome Odunze|WR|CHI","Jalen Hurts|QB|PHI","Jameson Williams|WR|DET","Mike Evans|WR|SF"],
["Rhamondre Stevenson|RB|NE","Davante Adams|WR|LAR","Tucker Kraft|TE|GB","Parker Washington|WR|JAX","Christian Watson|WR|GB","Justin Herbert|QB|LAC","Marvin Harrison Jr.|WR|ARI","Carnell Tate|WR|TEN","Drake Maye|QB|NE","Caleb Williams|QB|CHI","TreVeyon Henderson|RB|NE","Josh Downs|WR|IND"],
["MarShawn Lloyd|RB|GB","DK Metcalf|WR|PIT","Quentin Johnston|WR|LAC","Stefon Diggs|WR|WAS","Jaylen Warren|RB|PIT","Tony Pollard|RB|TEN","Jonathon Brooks|RB|CAR","Jayden Daniels|QB|WAS","Dak Prescott|QB|DAL","Rico Dowdle|RB|PIT","Harold Fannin Jr.|TE|CLE","Trevor Lawrence|QB|JAX"],
["Blake Corum|RB|LAR","Brock Purdy|QB|SF","Chuba Hubbard|RB|CAR","RJ Harvey|RB|DEN","J.K. Dobbins|RB|DEN","Chris Godwin Jr.|WR|TB","Jordan Mason|RB|MIN","Brian Thomas Jr.|WR|JAX","KC Concepcion|WR|CLE","Courtland Sutton|WR|DEN","Jordan Addison|WR|MIN","Dalton Kincaid|TE|BUF"],
["Michael Pittman Jr.|WR|PIT","Makai Lemon|WR|PHI","Jacory Croskey-Merritt|RB|WAS","Kenny Gainwell|RB|TB","Kyle Pitts Sr.|TE|ATL","Alec Pierce|WR|IND","Aaron Jones Sr.|RB|MIN","George Kittle|TE|SF","Josh Jacobs|RB|GB","Isaiah Likely|TE|NYG","Michael Wilson|WR|ARI","Jayden Reed|WR|GB"],
["Kyle Monangai|RB|CHI","Bo Nix|QB|DEN","Wan'Dale Robinson|WR|TEN","Matthew Golden|WR|GB","Travis Kelce|TE|KC","Brandon Aubrey|K|DAL","Rams|DEF|LAR","De'Zhaun Stribling|WR|SF","Mike Washington Jr.|RB|LV","Woody Marks|RB|HOU","Cameron Dicker|K|LAC","Isiah Pacheco|RB|DET"],
["Jalen Coker|WR|CAR","Matthew Stafford|QB|LAR","Xavier Worthy|WR|KC","Romeo Doubs|WR|NE","Texans|DEF|HOU","Tyjae Spears|RB|TEN","Rachaad White|RB|WAS","Kyler Murray|QB|MIN","Mark Andrews|TE|BAL","Ka'imi Fairbairn|K|HOU","Chris Rodriguez Jr.|RB|JAX","Patrick Mahomes|QB|KC"],
["Broncos|DEF|DEN","Dallas Goedert|TE|PHI","Deebo Samuel Sr.|WR|SF","Khalil Shakir|WR|BUF","Keaton Mitchell|RB|LAC","Tyler Allgeier|RB|ARI","Jake Ferguson|TE|DAL","Brian Robinson|RB|ATL","Seahawks|DEF|SEA","Jaxson Dart|QB|NYG","Alvin Kamara|RB|NO","Jared Goff|QB|DET"],
["Jordyn Tyson|WR|NO","Najee Harris|RB|NYG","Eagles|DEF|PHI","Jason Myers|K|SEA","Jakobi Meyers|WR|JAX","Tank Bigsby|RB|PHI","Jonah Coleman|RB|DEN","Ravens|DEF|BAL","Oronde Gadsden|TE|LAC","Kayshon Boutte|WR|HOU","Jordan Love|QB|GB","Cam Little|K|JAX"],
["Zach Charbonnet|RB|SEA","Vikings|DEF|MIN","Cooper Kupp|WR|SEA","Patriots|DEF|NE","Tyler Loop|K|BAL","Juwan Johnson|TE|NO","Rashid Shaheed|WR|SEA","Harrison Mevis|K|LAR","Tyler Shough|QB|NO","Will Reichard|K|MIN","Kaelon Black|RB|SF","Jake Bates|K|DET"],
["Justice Hill|RB|BAL","Steelers|DEF|PIT","James Conner|RB|ARI","Hunter Henry|TE|NE","Ja'Kobi Lane|WR|BAL","Ray Davis|RB|BUF","Jalen McMillan|WR|TB","Denzel Boston|WR|CLE","Andy Borregales|K|NE","49ers|DEF|SF","Cairo Santos|K|CHI","Tre Tucker|WR|LV"]
];

// Build seat-indexed grid
const grid    = [];
const rosters = Array.from({length:12}, () => []);

ROUNDS.forEach((round, r) => {
  const row = new Array(12);
  round.forEach((raw, i) => {
    const [name, pos, team] = raw.split("|");
    const seat    = r % 2 === 0 ? i + 1 : 12 - i;
    const overall = r * 12 + i + 1;
    const pick    = {name, pos, team, overall, seat};
    row[seat - 1] = pick;
    rosters[seat - 1].push(pick);
  });
  grid.push(row);
});

// ── Build header ──
const hrow = document.getElementById("hrow");

// rail corner
const railTh = document.createElement("th");
railTh.className = "rail-th";
railTh.innerHTML = `<div class="th-seat" style="font-size:10px;color:var(--ink3)">RD</div>`;
hrow.appendChild(railTh);

TEAMS.forEach((team, i) => {
  const seat   = i + 1;
  const counts = {};
  rosters[i].forEach(p => counts[p.pos] = (counts[p.pos] || 0) + 1);
  const tally  = ORDER.filter(p => counts[p])
    .map(p => `<b class="${p}">${p}&nbsp;${counts[p]}</b>`).join("");

  const th = document.createElement("th");
  th.dataset.seat = seat;
  th.className = seat === MY_SEAT ? "mine-th" : "";
  th.setAttribute("role","button");
  th.setAttribute("tabindex","0");
  th.innerHTML = `<div class="th-seat">SEAT ${seat}</div>
    <div class="th-team">${team}</div>
    <div class="th-tally">${tally}</div>`;
  hrow.appendChild(th);
});

// ── Build rows ──
const tbody = document.getElementById("tbody");

grid.forEach((row, r) => {
  const tr = document.createElement("tr");

  const rail = document.createElement("td");
  rail.className = "rail";
  rail.innerHTML = `<div class="rd-n">${r+1}</div><div class="rd-dir">${r % 2 === 0 ? "→" : "←"}</div>`;
  tr.appendChild(rail);

  row.forEach(p => {
    const td = document.createElement("td");
    td.className = "pick-cell" + (p.seat === MY_SEAT ? " mine-cell" : "");
    td.dataset.seat = p.seat;

    const isMine = p.seat === MY_SEAT;
    td.innerHTML = `<div class="card ${p.pos}${isMine ? " mine-card" : ""}"
        data-pos="${p.pos}" data-name="${p.name.toLowerCase()}">
      <div class="c-no">${p.overall}</div>
      <div class="c-name">${p.name}</div>
      <div class="c-meta">
        <span class="c-pos ${p.pos}">${p.pos}</span>
        <span class="c-tm">${p.team}</span>
      </div>
    </div>`;
    tr.appendChild(td);
  });

  tbody.appendChild(tr);
});

// ── Filtering ──
let posFilter = "ALL";
let query     = "";
let spotlight = null;

const cards = [...document.querySelectorAll(".card")];
const hint  = document.getElementById("hint");

function apply() {
  let shown = 0;
  cards.forEach(el => {
    const ok = (posFilter === "ALL" || el.dataset.pos === posFilter)
            && (!query || el.dataset.name.includes(query));
    el.classList.toggle("muted", !ok);
    if (ok) shown++;
  });

  // spotlight
  document.querySelectorAll("td.pick-cell").forEach(td =>
    td.classList.toggle("spot-cell", spotlight !== null && +td.dataset.seat === spotlight));
  document.querySelectorAll("thead th[data-seat]").forEach(th =>
    th.classList.toggle("spot-th", spotlight !== null && +th.dataset.seat === spotlight));

  const parts = [`${shown} of 180 picks`];
  if (spotlight) parts.push(TEAMS[spotlight - 1]);
  hint.textContent = parts.join(" · ");
}

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    posFilter = chip.dataset.pos;
    document.querySelectorAll(".chip").forEach(c =>
      c.setAttribute("aria-pressed", String(c === chip)));
    apply();
  });
});

document.getElementById("search").addEventListener("input", e => {
  query = e.target.value.trim().toLowerCase();
  apply();
});

function toggleSeat(seat) {
  spotlight = spotlight === seat ? null : seat;
  apply();
}

document.querySelectorAll("thead th[data-seat]").forEach(th => {
  th.addEventListener("click", () => toggleSeat(+th.dataset.seat));
  th.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleSeat(+th.dataset.seat); }
  });
});

apply();
</script>
</body>
</html>
