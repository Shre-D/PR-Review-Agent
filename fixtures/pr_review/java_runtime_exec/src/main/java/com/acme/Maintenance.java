package com.acme;

public class Maintenance {
    public void runScript(String cmd) throws Exception {
        Runtime.getRuntime().exec(cmd);
    }
}
