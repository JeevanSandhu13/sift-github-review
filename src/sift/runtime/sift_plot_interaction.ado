*! version 0.0.1  Sift runtime: predicted-response curve from e().
*!
*! After a regression, plot the predicted response across the
*! observed range of one variable, holding the other predictors at
*! their means. Uses Stata's native ``margins`` + ``marginsplot``.
*!
*! Usage:
*!   regress log_salary female log_assets log_age tenure
*!   sift_plot_interaction log_assets, label("Salary by log assets")
*!
*!   * Optional ``xlabel`` / ``ylabel`` / ``title`` to override defaults:
*!   sift_plot_interaction log_assets, ///
*!       label("Salary by log assets") ///
*!       xlabel("Log assets (USD)") ylabel("Log CEO salary") ///
*!       title("Predicted CEO salary by firm size")
*!
*! Why this exists: without a Stata helper, the model would extract
*! coefficients, switch to R or Python, and hand-roll a predicted-
*! response curve there — exactly the language-switching loop the
*! researcher complained about. Same posture as
*! ``sift_plot_coefficients``: native to Stata, manifest-allowlisted.

program define sift_plot_interaction
    version 13
    syntax varname , [ label(string) xlabel(string) ylabel(string) title(string) ]

    if "`e(cmd)'" == "" {
        display as error "sift_plot_interaction: no estimation results in memory. Run a regression first."
        exit 198
    }

    local var "`varlist'"

    * Resolve run dir from SIFT_RESULT_PATH (Stata's subprocess cwd
    * is the SESSION cwd via the executor preamble cd, NOT the
    * run dir).
    local resultpath : env SIFT_RESULT_PATH
    if "`resultpath'" == "" {
        display as error "sift_plot_interaction: SIFT_RESULT_PATH not set"
        exit 198
    }
    local rundir : subinstr local resultpath "/result.json" ""

    * Per-run authenticity token. Stamped into every manifest line so
    * the executor can drop manifest entries a hand-crafted file write
    * could otherwise have appended (a script saving a raw-data plot
    * under _sift_plots/ and labeling it ``interaction`` would
    * otherwise ride the next turn through the vision channel).
    local _sift_token : env SIFT_RUN_TOKEN
    if "`_sift_token'" == "" {
        display as error "sift_plot_interaction: SIFT_RUN_TOKEN not set"
        exit 198
    }

    local _step "init"
    capture noisily {
        local _step "summarize"
        * Compute the variable's mean and SD — NOT min/max. The
        * rendered PNG is allowlisted for model vision, so anything
        * the x-axis exposes crosses the SDC boundary; raw extrema
        * are exactly the disclosure the descriptive sanitizer
        * refuses unless the researcher has opted the variable into
        * ``non_disclosive_variables``. Mean ± 2*SD is equivalent to
        * the mean+SD pair the descriptive sanitizer already permits.
        quietly summarize `var'
        if r(N) < 10 {
            display as error "sift_plot_interaction: `var' has fewer than 10 non-missing observations; below disclosure threshold"
            exit 198
        }
        if r(sd) == 0 | missing(r(sd)) {
            display as error "sift_plot_interaction: `var' has zero variance — interaction plot would expose the constant value"
            exit 198
        }
        local _lower = r(mean) - 2 * r(sd)
        local _upper = r(mean) + 2 * r(sd)

        * Build a 25-point grid across the disclosure-safe range. 25 is
        * smooth enough for ``marginsplot`` 's recast(line) without
        * burning compute on huge datasets.
        local step = (`_upper' - `_lower') / 24
        margins, at(`var' = (`_lower'(`step')`_upper')) atmeans

        * Pick defaults for axis labels / title that are honest if
        * the caller doesn't override them. Match the R / Python
        * helpers' phrasing so plots from the three languages share
        * a vocabulary.
        local _xt `"`xlabel'"'
        if "`_xt'" == "" local _xt "`var'"
        local _yt `"`ylabel'"'
        if "`_yt'" == "" local _yt "Predicted response"
        local _tt `"`title'"'
        if "`_tt'" == "" local _tt "Predicted response by `var'"

        * recast(line) for the prediction line, recastci(rarea) for
        * a filled CI band — much more readable than the default
        * dashed-line whiskers.
        marginsplot, ///
            recast(line) recastci(rarea) ///
            plotopts(lcolor(navy) lwidth(medthick)) ///
            ciopts(color(navy%18)) ///
            title(`"`_tt'"', size(medium)) ///
            xtitle(`"`_xt'"') ytitle(`"`_yt'"') ///
            xlabel(, labsize(small)) ylabel(, angle(0) labsize(small)) ///
            graphregion(color(white)) plotregion(margin(medsmall))

        local _step "export"
        * Sanitize the variable name into a filename-safe token.
        local safe : subinstr local var ":" "_", all
        local safe : subinstr local safe "." "_", all
        _sift_export_plot using "`rundir'/_sift_plots", ///
            basename("interaction_`safe'") width(1600)
        local _file = "`r(file)'"
        local _fmt  = "`r(format)'"
        if "`_file'" == "" {
            display as error "sift_plot_interaction: every export format failed (last _rc=`r(last_rc)')"
            exit `r(last_rc)'
        }

        local _step "manifest"
        local lab `"`label'"'
        if "`lab'" == "" local lab "Predicted response by `var'"
        local lab : subinstr local lab "\" "\\", all
        local lab : subinstr local lab `"""' `"\""', all
        * RFC 8259 §7 forbids raw U+0000..U+001F inside JSON strings;
        * `json.loads` rejects the line and the executor's parser drops
        * the whole payload silently. Replace every control char with
        * a space (same posture as the original \t/\n/\r handling) so
        * a label byte from a Stata-imported automated-export dataset
        * can't make the manifest line invalid.
        forvalues _cc = 1/31 {
            if `_cc' != 9 & `_cc' != 10 & `_cc' != 13 {
                local lab : subinstr local lab "`=char(`_cc')'" " ", all
            }
        }
        local lab : subinstr local lab "`=char(127)'" " ", all
        local lab : subinstr local lab "`=char(10)'" " ", all
        local lab : subinstr local lab "`=char(13)'" " ", all
        local lab : subinstr local lab "`=char(9)'" " ", all

        local manifestpath "`rundir'/_sift_plots/manifest.jsonl"
        local jsonline `"{"file":"`_file'","kind":"interaction","label":"`lab'","format":"`_fmt'","_token":"`_sift_token'"}"'
        tempname mh
        file open `mh' using "`manifestpath'", write append text
        file write `mh' `"`jsonline'"' _n
        file close `mh'

        display as text "sift_plot_interaction: wrote `_file'"
    }

    if _rc {
        local _orig_rc = _rc
        capture mkdir "`rundir'/_sift_plots"
        local errpath "`rundir'/_sift_plots/helper_errors.jsonl"
        local errline `"{"helper":"plot_interaction","step":"`_step'","error":"Stata _rc=`_orig_rc'","message":"plot_interaction failed at step `_step' with _rc=`_orig_rc'; check stderr.log for the underlying error"}"'
        tempname eh
        capture file open `eh' using "`errpath'", write append text
        if !_rc {
            file write `eh' `"`errline'"' _n
            file close `eh'
        }
    }
end
