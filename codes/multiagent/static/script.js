// static/script.js
function runDebate(dialogue) {
  $("#chatlog").html("");
  addMessage("system", "Starting debate between Agent A and Agent B...");

  // ---- Step 1: Agent A ----
  $.ajax({
    url: "/debate_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({ dialogue, step: "A" }),
    success: function (resA) {
      addMessage("agentA", "Agent A Analysis:\n" + JSON.stringify(resA.analysis, null, 2));
      addMessage("system", "Agent B analyzing...");

      // ---- Step 2: Agent B ----
      $.ajax({
        url: "/debate_step",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ dialogue, step: "B" }),
        success: function (resB) {
          addMessage("agentB", "Agent B Analysis:\n" + JSON.stringify(resB.analysis, null, 2));
          addMessage("system", "Agents debating final result...");

          // ---- Step 3: Final debate ----
          $.ajax({
            url: "/debate_step",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ dialogue, step: "final" }),
            success: function (final) {
              addMessage("system", "🏁 Debate concluded!");
              addMessage("system", JSON.stringify(final.result, null, 2));
            },
            error: function () {
              addMessage("system", "❌ Error in final debate.");
            }
          });
        },
        error: function () {
          addMessage("system", "❌ Error in Agent B.");
        }
      });
    },
    error: function () {
      addMessage("system", "❌ Error in Agent A.");
    }
  });
}

function addMessage(role, text) {
  const div = $("<div></div>").addClass("msg").addClass(role).text(text);
  $("#chatlog").append(div);
  $("#chatlog").scrollTop($("#chatlog")[0].scrollHeight);
}
