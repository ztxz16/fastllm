import {mountSessionAgent} from "../../plugin-core/session-agent.js";

export const mountClaude = options => mountSessionAgent({...options, id:"claude", name:"Claude Code"});
