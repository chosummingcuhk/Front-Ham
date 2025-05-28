# Front Ham
### Taskmaster S19E4 Live Task _Extended_ Rules

## **Front Ham Game Rules**  

---

### **General Principles** 
**1. Self-elimination**
- **Not** prohibited.

**2. Introduction of `No-Op`**
- `No-Op`s (no-operation) are introduced for programming convenience.
- See appendix for details.

**3. Invalid Moves**  
- Any removal that would result in **negative sock counts** at any point is **invalid**.
- Any sock additions to **eliminated players** at any point is **invalid**.
  - It implies that no revival is possible after a player has been eliminated.

---

#### **Standard Play (More Than 2 Players)**  
**1. Turns & Actions**  
- Players take turns in a **fixed, randomized order**.  
- Each player must perform an action, comprised of:  
  - **Removals**: 3 valid removals.  
  - **Addition**: 1 valid addition.  
  - Example: `[Red, Blue, Green]` + `Yellow`.  

**2. Elimination**  
- **Checked and effected after the full action (removals + addition)**.  
- If a player’s sock count is **0 at the end of any player's turn**, they are **eliminated permanently**.

---

#### **Final Phase (Endgame: 2 Players Left)**  
**1. Turns & Actions**
- The two remaining Players alternate, in accordance with the randomized order established previously.
- Players can only use `No-Op`s when it is possible to eliminate a player with some (one or two) removals, and they wish to end the game.  

**2. Immediate Elimination (sudden death)**
- **This rule will only be applied if a valid action containing `No-Op`(s) have been submitted.** Otherwise, the elimination rule in **standard play** is used.
- The game will end once a player's sock reaches 0.

---

### **Appendix**  
- As `No-Op`s are introduced for convenience, their use mirrors the constraints of the physical game.
- They are not intended to substantively alter the rules of the game.
- **Valid `No-Op`-containing actions**:  
  - `[Color, No-Op, No-Op] + No-Op`: Remove 1 sock of the specified color.  
  - `[Color, Color, No-Op] + No-Op`: Remove 2 socks of the specified colors.
  - `[Color, Color, Color] + No-Op`: Remove 3 socks of the specified colors. **Allowed only when the player wins the game through this move.**  
- **Invalid `No-Op`-containing actions**:  
  - `[No-Op, No-Op, No-Op] + No-Op` must remove at least 1 sock. 
  - `[No-Op, No-Op, No-Op] + Color` the use of `No-Op`s requires forgoing adding 1 sock.
  - `[Color, No-Op, No-Op] + Color`
  - `[Color, Color, No-Op] + Color`
  - `[Color, Color, Color] + Color`