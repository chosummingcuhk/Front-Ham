# Front-Ham
Taskmaster S19E4 Live Task

### **Front Ham Game Rules**  

---

#### **Standard Play (More Than 2 Players)**  
**1. Turns & Actions**  
- Players take turns in a **fixed, randomized order**.  
- Each player must submit an action:  
  - **Removals**: 3 valid removals (colors only, no `No-Op`).  
  - **Addition**: Add 1 sock to any color (including eliminated colors).  
  - Example: `[Red, Blue, Green]` + `Yellow`.  

**2. Invalid Moves**  
- Any action that would result in **negative sock counts** at any point is **invalid**.  
  - Example: Attempting to remove 2 socks from a color with only 1 sock → invalid.  

**3. Elimination**  
- **Checked after the full action (removals + addition)**.  
- If a player’s sock count is **0 at the end of their turn**, they are **eliminated permanently**.  

**4. Revival**  
- **No revival**: Once eliminated, players stay out permanently.  

---

#### **Final Phase (Endgame: 2 Players Left)**  
**1. Turns & Actions**  
- Players can use **restricted No-Op actions** to terminate removal processing early.  
- **Valid No-Op Formats**:  
  - `[Color, No-Op, No-Op]`: Remove 1 sock from a single color, then stop.  
  - `[Color, Color, No-Op]`: Remove 2 socks from a single color, then stop.  
- **All-No-Op actions are disallowed** (must remove at least 1 sock).  
- **Addition Step**:  
  - If **No-Op is used**, the **addition step is skipped entirely**.  
  - If **no No-Op is used**, proceed with normal addition.  

**2. Invalid Moves**  
- Any action that would result in **negative sock counts** at any point is **invalid**.  
  - Example: Attempting to remove 2 socks from a color with 1 sock → invalid.  
- Invalid removals (e.g., targeting 0-sock colors) are ignored (treated as `No-Op`).  

**3. Immediate Elimination Rule**  
- **If No-Op is used in removals**:  
  - If any removal step reduces a player’s sock count to **0**, the game ends **immediately**.  
  - **Addition step is skipped**, so revival is **impossible**.  
- **If No-Op is not used**:  
  - Elimination is checked after each removal step.  
    - If a player’s sock count drops to 0:  
      - If **addition color ≠ eliminated player’s color** → Game ends immediately.  
      - If **addition color = eliminated player’s color** → Game continues (elimination checked at end of turn).  

**4. Revival**  
- **Only possible if No-Op is not used and addition color matches the eliminated player’s color**.  
  - Example: Player B removes Player A’s last sock (A=0), then adds to A (A=1) → A survives.  
- **Self-Revival**:  
  - A player can revive themselves by reducing their sock count to 0 and restoring it via addition.  
  - Example: Player B (1 sock) removes it (0), then adds back (1) → survives.  

**5. Self-Elimination**  
- If a player removes all their socks and does **not add back**, they are eliminated.  

---

### **Key Mechanics Recap**  
- **Standard Play**:  
  - All actions must be 3 removals + 1 addition.  
  - No negative sock counts allowed.  
  - Eliminations checked at end of turn.  
- **Final Phase**:  
  - Restricted No-Op actions allowed.  
  - **Immediate Elimination**:  
    - Triggered by No-Op + sock count = 0 (addition skipped).  
    - Triggered by removals (no No-Op) + sock count = 0 + addition ≠ eliminated player’s color.  
  - **Delayed Elimination**:  
    - Removals (no No-Op) + sock count = 0 + addition = eliminated player’s color → survival if restored.  
  - **Revival**: Only possible without No-Op and with matching addition color.  

---

### **Example Scenarios**  

#### **Scenario 1: Immediate Elimination with No-Op**  
- **Initial State**: Player A (1 sock), Player B (acting).  
- **Action**:  
  - Removals: `[A, No-Op, No-Op]` (No-Op used → addition skipped).  
- **Result**:  
  - A’s sock count = 0 → **A eliminated immediately** (addition skipped).  

#### **Scenario 2: Opponent Revival Without No-Op**  
- **Initial State**: Player A (1 sock), Player B (acting).  
- **Action**:  
  - Removals: `[A, A, A]` (no No-Op → addition allowed).  
  - Addition: `A`.  
- **Result**:  
  - A’s sock count: 1 → 0 → 1 → **A survives**.  

#### **Scenario 3: Invalid Action (Negative Socks)**  
- **Initial State**: Player A (1 sock), Player B (acting).  
- **Action**:  
  - Removals: `[A, A, A]` (invalid: removing 3 socks from A with 1 sock).  
- **Result**:  
  - Action is **invalid** → rejected.  

#### **Scenario 4: Strategic No-Op**  
- **Initial State**: Player A (2 socks), Player B (acting).  
- **Action**:  
  - Removals: `[A, A, No-Op]` (No-Op used → addition skipped).  
- **Result**:  
  - A’s sock count = 0 → **A eliminated immediately** (addition skipped).  

---

### **Why This Works**  
- **Clarity**: Rules cleanly divided into Standard Play and Final Phase.  
- **Balance**: No-Op adds strategic depth but sacrifices revival opportunities.  
- **Invalid Move Handling**: Prevents negative sock counts, ensuring valid gameplay.  
- **Tactical Final Phase**: Players must weigh risks of aggressive No-Op tactics vs. revival-focused strategies.