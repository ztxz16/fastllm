import {mountSessionAgent} from "../../plugin-core/session-agent.js";

export const mountCodex = options => mountSessionAgent({...options, id:"codex", name:"Codex"});
