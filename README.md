# Front Ham
### Taskmaster S19E4 Live Task _Extended_ Rules

## **Front Ham Game Rules**  

---

### **General Overview**  
- **Players**: 5 players, each secretly assigned a unique color (e.g. Red, Blue, Green, Yellow, Purple).  
- **Socks**: 6 socks per color are publicly visible on a clothesline (all players know the current sock counts for every color).  
- **Goal**: Eliminate players by reducing their color’s sock count to 0. The last player remaining wins.  
- **Sock Counts**: All players know how many socks remain for every color (public information).  
- **Known**: Players can deduce who was eliminated by tracking which color’s sock count reached 0.  
- **Unknown**: Players do not know which other player corresponds to which color.  
---

### **General Principles** 
**1. Self-elimination**
- **Not** prohibited.

**2. Introduction of `No-Op`**
- `No-Op`s (no-operation) are introduced for programming convenience.
- See appendix for details.

**3. Validity of Moves**  
- Any removal that would result in **negative sock counts** at any point is **invalid**.
- Any sock additions to **eliminated players** at any point is **invalid**.
  - It implies that no revival is possible after a player has been eliminated.

---
### **Order of Play**  
- Players take turns in a **fixed, randomized order** (decided at the start).  
- Turns proceed until only **2 players remain**, after which the game enters the **Final Phase**.  
---

### **Standard Play (More Than 2 Players)**  
**1. Valid Actions**
- Each player must perform a valid action, comprised of:  
  - **Removals**: 3 valid removals.  
  - **Addition**: 1 valid addition.  
  - Example: `[Red, Blue, Green]` + `Yellow`.  
  - **No No-Op actions allowed** in standard play.  
    
**2. Elimination**  
- **Checked and effected after the full action (removals + addition)**.  
- If a player’s sock count is **0 at the end of any player's turn**, they are **eliminated permanently**.

---

### **Final Phase (Endgame: 2 Players Left)**  
**1. Turns & Actions**
- The two remaining Players alternate, in accordance with the randomized order established previously.
- `No-Op`s are now allowed, but only specific patterns are valid (see **Appendix**).  
- Players can only use `No-Op`s when it is possible to eliminate a player with some removals, and they wish to end the game.  

**2. Immediate Elimination (sudden death)**
- **This rule will only be applied if a valid action containing `No-Op`(s) have been submitted.** Otherwise, the elimination rule in **standard play** is used.
- The game will end once a player's sock count reaches 0.

**3. Validity of Actions** 
 - **Actions Without No-Ops**: Must satisfy the rule stated in **Valid Actions** under **Standard Play**.
  - **Actions With No-Ops**: Must satisfy the requirements stated in the **Appendix** (e.g., removing 1 sock with `[Red, No-Op, No-Op]`).  

---

### **Appendix**  
- As `No-Op`s are introduced for convenience, their use mirrors the constraints of the physical game. They are not intended to substantively alter the rules of the game.
- **They are allowed only when they result in the elimination of a player.** Due to this restriction, only one color can be specified in the move.
- **Valid `No-Op`-containing actions**:  
  - `[Color, No-Op, No-Op] + No-Op`: Remove 1 sock of the specified color.  
  - `[Color, Color, No-Op] + No-Op`: Remove 2 socks of the specified color.
  - `[Color, Color, Color] + No-Op`: Remove 3 socks of the specified color.   
- **Invalid `No-Op`-containing actions**:  
  - Any other `No-Op`-containing action.

  
